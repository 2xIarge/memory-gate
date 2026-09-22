"""Test suite for memory-gate.

Two of these tests exist to protect an assumption rather than a behaviour:
``before_model`` being its own graph node, and ``wrap_model_call`` composing
first-outermost. The whole design leans on that factory internals layout; if
LangChain changes it, these fail loudly instead of the gate silently stopping
working.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelFallbackMiddleware,
    ModelRetryMiddleware,
    SummarizationMiddleware,
)
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from memory_gate import Archive, ArchiveWriteError, GateError, MemoryGateMiddleware, Record
from memory_gate.middleware import _kind_of
from memory_gate.review import parse_reply

FISCAL = "By the way, our fiscal year starts April 1st, so Q1 is Apr-Jun."
PROBE = "fiscal year starts April"
LOSSY_SUMMARY = "## SESSION INTENT\nAnalyse quarterly revenue.\n## SUMMARY\nQueries were run."


@pytest.fixture()
def workdir():
    path = Path(tempfile.mkdtemp(prefix="mg-test-"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


def history(n_filler: int = 6) -> list[tuple[str, str]]:
    msgs = [("user", "Analyse our quarterly revenue for 2026."), ("assistant", "Which calendar?")]
    msgs.append(("user", FISCAL))
    for i in range(n_filler):
        msgs.append(("user", f"run query {i}"))
        msgs.append(("assistant", f"query {i} done"))
    msgs.append(("user", "Now give me the Q1 chart."))
    return msgs


class Recorder(AgentMiddleware):
    """Captures the full request payload, not just the system message."""

    def __init__(self) -> None:
        super().__init__()
        self.last_system = ""
        self.last_body = ""
        self.last_all = ""

    def wrap_model_call(self, request, handler):  # noqa: ANN001
        sm = getattr(request, "system_message", None)
        self.last_system = str(sm.content) if sm is not None else ""
        self.last_body = "\n".join(str(getattr(m, "content", "")) for m in request.messages)
        self.last_all = f"{self.last_system}\n{self.last_body}"
        return handler(request)


def fake_model(text: str, repeats: int = 80) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([AIMessage(content=text)] * repeats))


def make_summarizer() -> SummarizationMiddleware:
    return SummarizationMiddleware(
        model=fake_model(LOSSY_SUMMARY), trigger=("messages", 10), keep=("messages", 4)
    )


def drive(agent, thread: str, msgs):  # noqa: ANN001
    """Run to completion, answering each gate interrupt by pinning the probe line."""
    interrupts: list[dict] = []
    config = {"configurable": {"thread_id": thread}}
    for message in msgs:
        result = agent.invoke({"messages": [message]}, config=config)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            interrupts.append(payload)
            items = payload["items"]
            assert not [i for i in items if "summary of the conversation" in i["preview"].lower()]
            target = next((i["ref"] for i in items if PROBE in i["preview"]), None)
            reply = f"keep {target}" if target is not None else "confirm"
            result = agent.invoke(Command(resume=reply), config=config)
    return interrupts


# --------------------------------------------------------------- archive


def test_archive_is_idempotent(workdir: Path) -> None:
    archive = Archive(workdir)
    records = [Record(id=f"m{i}", role="human", turn=i, ts="t", tokens=5, text=f"x{i}")
               for i in range(4)]
    assert archive.append(records) == 4
    assert archive.append(records) == 0
    assert archive.seen_ids() == {r.id for r in records}


def test_archive_raises_when_unwritable(workdir: Path) -> None:
    blocker = workdir / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    archive = Archive(blocker / "nested")
    with pytest.raises(ArchiveWriteError):
        archive.append([Record(id="a", role="human", turn=1, ts="t", tokens=1, text="x")])


def test_protect_and_restore_roundtrip(workdir: Path) -> None:
    archive = Archive(workdir)
    archive.append([Record(id="a", role="human", turn=1, ts="t", tokens=3,
                           text="fx rate rule", kind="dialogue")])
    gate = MemoryGateMiddleware(archive_dir=workdir, trigger=("messages", 2))
    assert gate.restore("a").text == "fx rate rule"
    assert gate.archive.protected_ids() == {"a"}
    with pytest.raises(GateError):
        gate.restore("missing")


# ------------------------------------------------------------ reply parsing


@pytest.mark.parametrize(
    ("text", "mode", "refs"),
    [
        ("keep 1,3,5", "keep_selected", [1, 3, 5]),
        ("保留 2,4", "keep_selected", [2, 4]),
        ("keep: 1 3", "keep_selected", [1, 3]),
        ("3", "keep_selected", [3]),
        ("keep all", "keep_all", [1, 2, 3, 4, 5]),
        ("全部保留", "keep_all", [1, 2, 3, 4, 5]),
        ("confirm", "confirm", []),
        ("确认", "confirm", []),
        # anything ambiguous must keep everything rather than guess
        ("keep the flower one", "unrecognized", []),
        ("", "unrecognized", []),
        ("keep 99", "unrecognized", []),
        # a partially-valid request is still ambiguous: do not honour half of it
        ("keep 2,9", "unrecognized", []),
    ],
)
def test_parse_reply(text: str, mode: str, refs: list[int]) -> None:
    valid = {1, 2, 3, 4, 5}
    got = parse_reply(text, valid)
    assert got["mode"] == mode
    assert got["keep_refs"] == (sorted(valid) if mode == "keep_all" else refs)


def test_structured_reply_with_unknown_ref_fails_closed() -> None:
    got = parse_reply({"mode": "keep_selected", "keep_refs": [1, 77]}, {1, 2, 3})
    assert got["mode"] == "unrecognized"
    assert got["keep_refs"] == []


def test_unrecognized_reply_pins_everything_end_to_end(workdir: Path) -> None:
    """A junk resume value must never silently drop something."""
    gate = MemoryGateMiddleware(archive_dir=workdir, trigger=("messages", 3),
                                keep_messages=1, review_threshold_tokens=0)
    agent = create_agent(model=fake_model("ok"), tools=[],
                         middleware=[gate, make_summarizer()],
                         checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "junk"}}
    result = None
    for message in history(2):
        result = agent.invoke({"messages": [message]}, config=config)
        if "__interrupt__" in result:
            result = agent.invoke(Command(resume="hmm let me think about it"), config=config)
    stats = gate.archive.stats()
    assert stats["protected"] >= 1
    assert stats["protected"] <= stats["archived"]


# ------------------------------------------------------ factory assumptions


def test_before_model_runs_in_list_order() -> None:
    order: list[str] = []

    def probe(label: str):
        return type(
            f"Probe_{label}",
            (AgentMiddleware,),
            {"before_model": lambda self, state, runtime: (order.append(label), None)[1]},
        )()

    agent = create_agent(model=fake_model("ok", 6), tools=[],
                         middleware=[probe("a"), probe("b"), probe("c")])
    agent.invoke({"messages": [("user", "hi")]})
    assert order[:3] == ["a", "b", "c"]


def test_wrap_model_call_first_is_outermost() -> None:
    order: list[str] = []

    def wrapper(label: str):
        def wrap(self, request, handler):  # noqa: ANN001
            order.append(f"in:{label}")
            res = handler(request)
            order.append(f"out:{label}")
            return res

        return type(f"Wrap_{label}", (AgentMiddleware,), {"wrap_model_call": wrap})()

    agent = create_agent(model=fake_model("ok", 6), tools=[],
                         middleware=[wrapper("outer"), wrapper("inner")])
    agent.invoke({"messages": [("user", "hi")]})
    assert order[:4] == ["in:outer", "in:inner", "out:inner", "out:outer"]


def test_interrupt_in_before_model_survives_retry_and_fallback() -> None:
    class Interrupter(AgentMiddleware):
        def before_model(self, state, runtime):  # noqa: ANN001
            from langgraph.types import interrupt
            interrupt({"type": "probe"})
            return None

    model = fake_model("ok", 8)
    agent = create_agent(
        model=model,
        tools=[],
        middleware=[Interrupter(), ModelRetryMiddleware(max_retries=2), ModelFallbackMiddleware(model)],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "int-safe"}}
    result = agent.invoke({"messages": [("user", "hi")]}, config=config)
    assert "__interrupt__" in result, "retry/fallback swallowed the interrupt"
    resumed = agent.invoke(Command(resume="pong"), config=config)
    assert "__interrupt__" not in resumed
    text = " ".join(str(getattr(m, "content", "")) for m in resumed["messages"])
    assert "probe" not in text, "interrupt leaked into the transcript as an AIMessage"


# ------------------------------------------------------------ end to end


def test_without_gate_compaction_loses_the_constraint() -> None:
    recorder = Recorder()
    agent = create_agent(model=fake_model("ok"), tools=[],
                         middleware=[recorder, make_summarizer()], checkpointer=InMemorySaver())
    for message in history():
        agent.invoke({"messages": [message]}, config={"configurable": {"thread_id": "no-gate"}})
    assert PROBE not in recorder.last_all, (
        "compaction did not run; this test is only meaningful if it destroys the line"
    )


def test_with_gate_the_constraint_is_injected_not_lucky(workdir: Path) -> None:
    recorder = Recorder()
    summarizer = make_summarizer()
    gate = MemoryGateMiddleware(archive_dir=workdir, summarization=summarizer,
                                review_threshold_tokens=1, protect_patterns=(r"fiscal year",))
    agent = create_agent(model=fake_model("ok"), tools=[],
                         middleware=[gate, recorder, summarizer], checkpointer=InMemorySaver())
    interrupts = drive(agent, "with-gate", history())

    assert interrupts, "the gate never fired"
    assert PROBE not in recorder.last_body, "the message survived on its own; gate did nothing"
    assert PROBE in recorder.last_system, "gate did not inject the pinned line"


def test_gate_does_not_nag(workdir: Path) -> None:
    """Interrupts must track compactions, not turns.

    A gate that interrupts on nearly every turn gets switched off, which is the
    real-world failure mode. Trigger alignment brought 13-per-16-turns down to 2.
    """
    summarizer = make_summarizer()
    gate = MemoryGateMiddleware(archive_dir=workdir, summarization=summarizer,
                                review_threshold_tokens=1, protect_patterns=(r"fiscal year",))
    agent = create_agent(model=fake_model("ok"), tools=[],
                         middleware=[gate, summarizer], checkpointer=InMemorySaver())
    interrupts = drive(agent, "nag", history())
    turns = len(history())
    assert len(interrupts) <= 4, f"{len(interrupts)} interrupts across {turns} turns"


def test_every_gate_run_is_recorded(workdir: Path) -> None:
    summarizer = make_summarizer()
    gate = MemoryGateMiddleware(archive_dir=workdir, summarization=summarizer,
                                review_threshold_tokens=1, protect_patterns=(r"fiscal year",))
    agent = create_agent(model=fake_model("ok"), tools=[],
                         middleware=[gate, summarizer], checkpointer=InMemorySaver())
    drive(agent, "audit", history())
    lines = (workdir / "runs.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert lines, "no audit trail written"
    assert any('"reviewed": true' in line for line in lines)


# -------------------------------------------------------------- guardrails


def test_constructing_without_a_trigger_is_refused(workdir: Path) -> None:
    with pytest.raises(ValueError, match="silently useless"):
        MemoryGateMiddleware(archive_dir=workdir)


@pytest.mark.parametrize(
    ("trigger", "n_messages", "tokens", "imminent"),
    [
        (("messages", 10), 10, 0, True),
        (("messages", 10), 9, 0, False),
        (("tokens", 100), 3, 99, False),
        (("tokens", 100), 3, 100, True),
        ({"tokens": 100, "messages": 5}, 5, 99, False),
        ({"tokens": 100, "messages": 5}, 5, 100, True),
        ([{"tokens": 100, "messages": 5}, ("messages", 50)], 50, 0, True),
    ],
)
def test_trigger_semantics_match_the_library(workdir: Path, trigger, n_messages, tokens,
                                             imminent) -> None:
    gate = MemoryGateMiddleware(archive_dir=workdir, trigger=trigger)
    assert gate._compaction_is_imminent(n_messages, tokens) is imminent


def test_compaction_summaries_are_never_offered_for_review(workdir: Path) -> None:
    summary = HumanMessage(content="Here is a summary of the conversation to date:\n\nx",
                           additional_kwargs={"lc_source": "summarization"})
    assert _kind_of(summary) == "summary"
    assert _kind_of(HumanMessage(content="a plain user turn")) == "dialogue"

    # and a summary sitting in the window must not reach the reviewer's list
    summarizer = SummarizationMiddleware(
        model=fake_model(LOSSY_SUMMARY), trigger=("messages", 3), keep=("messages", 1)
    )
    gate = MemoryGateMiddleware(archive_dir=workdir, summarization=summarizer,
                               review_threshold_tokens=0)
    agent = create_agent(model=fake_model("ok"), tools=[],
                         middleware=[gate, summarizer], checkpointer=InMemorySaver())
    turns = [("user", FISCAL)] + [("user", f"q{i}") for i in range(6)]
    interrupts = drive(agent, "no-summary-review", turns)
    for payload in interrupts:
        previews = " ".join(i["preview"] for i in payload["items"])
        assert "summary of the conversation" not in previews.lower()

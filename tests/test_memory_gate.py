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
import warnings
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
    records = [Record(id=f"m{i}", role="human", pass_no=i, ts="t", tokens=5, text=f"x{i}")
               for i in range(4)]
    assert archive.append(records) == 4
    assert archive.append(records) == 0
    assert archive.seen_ids() == {r.id for r in records}


def test_archive_raises_when_unwritable(workdir: Path) -> None:
    blocker = workdir / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    archive = Archive(blocker / "nested")
    with pytest.raises(ArchiveWriteError):
        archive.append([Record(id="a", role="human", pass_no=1, ts="t", tokens=1, text="x")])


def test_protect_and_restore_roundtrip(workdir: Path) -> None:
    archive = Archive(workdir)
    archive.append([Record(id="a", role="human", pass_no=1, ts="t", tokens=3,
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


def test_assistant_restatements_are_not_pinned_by_default(workdir: Path) -> None:
    """The assistant echoing a rule back is redundant with the user's own wording
    and competes for the same protect budget. Measured: roughly half the pinned
    tokens were restatements, halving how many distinct constraints fit."""
    from langchain_core.messages import HumanMessage
    from langgraph.types import Command

    def run_and_keep_all(gate, thread, turns):
        agent = create_agent(model=fake_model("这条也必须遵守。"), tools=[],
                             middleware=[gate, make_summarizer()],
                             checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": thread}}
        for text in turns:
            res = agent.invoke({"messages": [HumanMessage(content=text)]}, config=config)
            while "__interrupt__" in res:
                res = agent.invoke(Command(resume="keep all"), config=config)

    turns = ["报表必须用 net_revenue。", "图表必须加季度号。", "金额必须是美元。"] * 6

    # keep_messages must match make_summarizer()'s keep, or the gate computes an
    # empty window and never reviews -- exactly the mis-binding the constructor
    # now warns about.
    narrow = MemoryGateMiddleware(archive_dir=workdir / "narrow",
                                  trigger=("messages", 10), keep_messages=4,
                                  review_threshold_tokens=0,
                                  protect_patterns=(r"必须",))
    run_and_keep_all(narrow, "pinroles-narrow", turns)
    pinned = narrow.archive.protected()
    assert pinned, "the user's own constraints should be pinned"
    assert all(r.role == "human" for r in pinned), "assistant reply leaked into pins"

    wide = MemoryGateMiddleware(archive_dir=workdir / "wide",
                                trigger=("messages", 10), keep_messages=4,
                                review_threshold_tokens=0,
                                protect_patterns=(r"必须",), pin_roles=("human", "ai"))
    run_and_keep_all(wide, "pinroles-wide", turns)
    wide_pinned = wide.archive.protected()
    assert any(r.role == "ai" for r in wide_pinned), "widening pin_roles did nothing"
    # the whole point: fewer, denser pins for the same conversation
    assert len(pinned) < len(wide_pinned)


def test_empty_pin_roles_is_refused(workdir: Path) -> None:
    with pytest.raises(ValueError, match="pin_roles"):
        MemoryGateMiddleware(archive_dir=workdir, trigger=("tokens", 500), pin_roles=())


def test_protect_budget_bounds_the_injection(workdir: Path) -> None:
    """Regression: the pinned text is injected on *every* model call, so an
    uncapped protected set raises reported token usage, trips the compaction
    trigger sooner, causes more reviews, which pin more. Measured at 2 -> 31
    compaction cycles. The budget stops the growth and the overflow is declared.
    """
    archive = Archive(workdir)
    for i in range(10):
        rec = Record(id=f"m{i}", role="human", pass_no=i, ts="t", tokens=200,
                     text=f"constraint number {i}", kind="dialogue")
        archive.append([rec])
        archive.protect([rec])

    gate = MemoryGateMiddleware(archive_dir=workdir, trigger=("tokens", 500),
                                protect_budget_tokens=500)
    kept, omitted = gate.protected_view()
    assert sum(r.tokens for r in kept) <= 500
    assert {r.id for r in kept} == {"m8", "m9"}, "newest pins must win"
    assert [r.id for r in kept] == ["m8", "m9"], "injection must read chronologically"
    assert len(omitted) == 8


def test_absurd_budget_is_refused(workdir: Path) -> None:
    """A budget too small to hold anything would leave the gate looking armed
    while injecting nothing -- the silent no-op class of langchain#39247."""
    with pytest.raises(ValueError, match="protect_budget_tokens"):
        MemoryGateMiddleware(archive_dir=workdir, trigger=("tokens", 500),
                             protect_budget_tokens=10)


def test_constructing_without_a_trigger_is_refused(workdir: Path) -> None:
    with pytest.raises(ValueError, match="silently useless"):
        MemoryGateMiddleware(archive_dir=workdir)


def msgs(n: int, reported: int | None = None) -> list:
    """A message list of length n; last AIMessage carries `reported` total tokens."""
    from langchain_core.messages import AIMessage, HumanMessage

    out = [HumanMessage(content=f"h{i}") for i in range(max(0, n - 1))]
    ai = AIMessage(content="a")
    if reported is not None:
        ai.usage_metadata = {"input_tokens": reported, "output_tokens": 0,
                             "total_tokens": reported}
    out.append(ai)
    return out


@pytest.mark.parametrize(
    ("trigger", "n_messages", "tokens", "reported", "imminent"),
    [
        (("messages", 10), 10, 0, None, True),
        (("messages", 10), 9, 0, None, False),
        (("tokens", 100), 3, 99, None, False),
        (("tokens", 100), 3, 100, None, True),
        ({"tokens": 100, "messages": 5}, 5, 99, None, False),
        ({"tokens": 100, "messages": 5}, 5, 100, None, True),
        ([{"tokens": 100, "messages": 5}, ("messages", 50)], 50, 0, None, True),
        # the summariser fires on approximate OR model-reported tokens; the gate
        # has to match both halves or it reviews too late and loses content
        (("tokens", 500), 6, 120, 900, True),
        (("tokens", 500), 6, 120, 100, False),
    ],
)
def test_trigger_semantics_match_the_library(workdir: Path, trigger, n_messages, tokens,
                                             reported, imminent) -> None:
    gate = MemoryGateMiddleware(archive_dir=workdir, trigger=trigger)
    assert gate._compaction_is_imminent(msgs(n_messages, reported), tokens) is imminent


def test_records_number_passes_not_stale_list_indices(workdir: Path) -> None:
    """`turn` used to be the index in the live list, which stops meaning anything
    once compaction pins that list near the keep window."""
    summarizer = make_summarizer()
    gate = MemoryGateMiddleware(archive_dir=workdir, summarization=summarizer,
                                review_threshold_tokens=0, protect_patterns=(r"zzz",))
    agent = create_agent(model=fake_model("ok"), tools=[],
                         middleware=[gate, summarizer], checkpointer=InMemorySaver())
    drive(agent, "passno", history())
    passes = {r.pass_no for r in gate.archive._read(gate.archive.archive_path)}
    assert passes and min(passes) >= 1
    assert len(passes) > 1, "pass counter never advanced; it is a constant again"
    assert not hasattr(gate.archive._read(gate.archive.archive_path)[0], "turn")



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


def _budget_warnings(gate_kwargs) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        MemoryGateMiddleware(**gate_kwargs)
    return [str(w.message) for w in caught if "protect_budget_tokens" in str(w.message)]


def test_budget_near_the_trigger_warns(workdir: Path) -> None:
    """Pinned text is re-sent and re-counted every call, so a budget comparable
    to the trigger re-arms the compaction it just survived. Measured: 2 cycles
    became 31. `keep all` is the review's default action, which makes this a
    configuration a user can fall into without meaning to."""
    msgs = _budget_warnings({"archive_dir": workdir, "trigger": ("tokens", 1_000),
                             "keep_messages": 4, "protect_budget_tokens": 500})
    assert msgs, "a budget at half the trigger passed silently"
    assert "31" in msgs[0], "the warning should carry the measured consequence"


def test_budget_well_under_the_trigger_is_quiet(workdir: Path) -> None:
    """The production shape -- 160K trigger, a few thousand tokens of pins --
    must not warn, or the warning is noise and gets ignored."""
    assert not _budget_warnings({"archive_dir": workdir, "trigger": ("tokens", 160_000),
                                 "keep_messages": 4, "protect_budget_tokens": 3_200})


def test_message_trigger_does_not_invent_a_token_budget(workdir: Path) -> None:
    """A message-count trigger has no token figure to compare against. Guessing
    one would warn about configurations that may be perfectly safe."""
    gate = MemoryGateMiddleware(archive_dir=workdir, trigger=("messages", 6),
                                keep_messages=4, protect_budget_tokens=5_000)
    assert gate._earliest_trigger_tokens() is None
    assert not _budget_warnings({"archive_dir": workdir, "trigger": ("messages", 6),
                                 "keep_messages": 4, "protect_budget_tokens": 5_000})


@pytest.mark.parametrize("trigger,expected", [
    (("tokens", 160_000), 3_200),   # 2% of a production window
    (("tokens", 2_500), 200),       # clamped up: 2% would hold nothing
    (("tokens", 500_000), 4_000),   # clamped down: 2% would re-arm the trigger
    (("messages", 6), 1_200),       # no token figure to scale from
])
def test_default_budget_scales_with_the_trigger(workdir: Path, trigger, expected) -> None:
    """A constant default was wrong at both ends: 1 200 tokens is ~half of a
    2 500-token trigger (which re-armed compaction every call) and 0.6% of a
    200K window (which cannot hold one session's rules)."""
    gate = MemoryGateMiddleware(archive_dir=workdir, trigger=trigger, keep_messages=4)
    assert gate.protect_budget_tokens == expected
    assert not _budget_warnings({"archive_dir": workdir, "trigger": trigger,
                                 "keep_messages": 4}), \
        "the shipped default must not trip the library's own budget warning"


def test_tiny_windows_warn_even_on_the_default(workdir: Path) -> None:
    """Below a ~2 000-token trigger the 200-token floor is already over 10%, so
    the default warns. That is not a bug to suppress: on a window that small no
    budget able to hold a single rule can also stay out of the trigger's way.
    Recording it here so the boundary is a decision rather than a surprise."""
    msgs = _budget_warnings({"archive_dir": workdir, "trigger": ("tokens", 500),
                             "keep_messages": 4})
    assert msgs, "a 200-token budget on a 500-token trigger should not pass quietly"
    assert "40%" in msgs[0]


DECLARATIVE_CONSTRAINTS = (
    "对了我们财年是从 4 月 1 号起的，你按 4 到 6 月当第一季度。",
    "这个项目的对接人是 Lena，后面邮件直接发她。",
    "Our fiscal year starts in April.",
    "We use UTC for all timestamps in the logs.",
    "The staging database is read-only, nobody writes to it.",
    "版本号按 SemVer 走。",
    "客户那边只认 PDF。",
    "每周三下午三点站会，别迟到。",
)

NOT_CONSTRAINTS = (
    "要不要一起去看电影？",
    "要不要试试换个写法？",
    "明天会降温吗，要不要给孩子加件外套？",
    "今天午饭吃啥？",
    "Which movie should I watch tonight?",
    # A probe question from the verification harness. `只回答` was added to the
    # pattern list after reading a transcript full of these, which made the
    # scorer's own questions look like standing rules -- a leak from the test
    # fixture into the shipped defaults, caught by rendering at scale.
    "我们的 Q1 覆盖哪几个月？只回答月份。",
    # Same shape as a declaration, but asking: "内部代号是什么" matches the
    # 内部 + copula pattern that catches "对接人是 Lena".
    "这个项目的内部代号是什么？只回答代号。",
    "对接人是谁？",
)


def test_declarative_constraints_are_flagged(workdir: Path) -> None:
    """The shipped pattern list was imperative-only (must/never/必须/一律) and
    scored recall 4/10 on the constraints planted in a recorded run, because
    people state conventions as facts far more often than as commands. These are
    the sentences it used to miss; `scripts/score_patterns.py` holds the corpora
    and the held-out numbers."""
    gate = MemoryGateMiddleware(archive_dir=workdir, trigger=("tokens", 160_000),
                                keep_messages=4)
    for text in DECLARATIVE_CONSTRAINTS:
        assert gate._is_risky(text), f"declarative constraint not flagged: {text}"


def test_questions_are_not_flagged_as_prohibitions(workdir: Path) -> None:
    """`不要` fired inside `要不要` and bare `别` fired on 别太/别的, which made
    ordinary questions look like standing rules -- 6 of the 10 flags in the
    recorded run were this. Dropping those patterns outright measured worse, so
    the ambiguous readings are excluded instead."""
    gate = MemoryGateMiddleware(archive_dir=workdir, trigger=("tokens", 160_000),
                                keep_messages=4)
    for text in NOT_CONSTRAINTS:
        assert not gate._is_risky(text), f"chatter flagged as a constraint: {text}"

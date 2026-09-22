"""Runnable demo. No API key, no network -- fake models stand in for a real one.

    python examples/fiscal_year_demo.py

The scenario: a finance agent is told on turn 3 that the company's fiscal year
starts April 1st. Twelve turns of query noise later, context compaction wipes
that turn and replaces it with a summary that does not mention it. The agent
then answers "Q1" with calendar January-March, and nothing anywhere reports an
error.

Part A runs the stock ``SummarizationMiddleware``. Part B adds
``MemoryGateMiddleware`` in front of it.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain.agents import create_agent  # noqa: E402
from langchain.agents.middleware import AgentMiddleware, SummarizationMiddleware  # noqa: E402
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from memory_gate import MemoryGateMiddleware  # noqa: E402

FISCAL = (
    "By the way, our fiscal year starts April 1st, so Q1 is Apr-Jun. "
    "All prior reports use that basis."
)

# The summary the model actually produces. Note what is missing: the fiscal
# calendar. It reads as a perfectly good summary.
LOSSY_SUMMARY = (
    "## SESSION INTENT\nAnalyse quarterly revenue for 2026.\n\n"
    "## SUMMARY\nUser requested quarterly revenue analysis; several SQL queries "
    "were run against the warehouse.\n\n"
    "## ARTIFACTS\ncharts/q0.png .. charts/q5.png\n\n"
    "## NEXT STEPS\nProduce the Q1 chart."
)

HISTORY: list[tuple[str, str]] = [
    ("user", "Analyse our quarterly revenue for 2026."),
    ("assistant", "Sure. Which fiscal calendar should I use?"),
    ("user", FISCAL),
    *[("user", f"run query {i}") for i in range(6)],
    *[("assistant", f"query {i} done, wrote charts/q{i}.png") for i in range(6)],
    ("user", "Now give me the Q1 revenue chart."),
]


class RecordWhatTheModelSees(AgentMiddleware):
    """Last resort: capture everything that actually reaches the model.

    Checking only the system message would be trivially true in Part A (there
    is no system message at all), so we capture the full request payload and
    assert absence across all of it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.last_system_message = ""
        self.last_request_text = ""
        self.last_body_text = ""
        self.last_message_count = 0

    def wrap_model_call(self, request, handler):  # noqa: ANN001
        sm = getattr(request, "system_message", None)
        self.last_system_message = str(sm.content) if sm is not None else ""
        body = [str(getattr(m, "content", "")) for m in request.messages]
        self.last_message_count = len(body)
        self.last_body_text = "\n".join(body)
        self.last_request_text = "\n".join([self.last_system_message, *body])
        return handler(request)


def build_middlewares(observer: RecordWhatTheModelSees, gate=None):  # noqa: ANN001
    summarizer = SummarizationMiddleware(
        model=GenericFakeChatModel(messages=iter([AIMessage(content=LOSSY_SUMMARY)] * 5)),
        trigger=("messages", 10),
        keep=("messages", 4),
    )
    stack: list = [observer]
    if gate is not None:
        stack.insert(0, gate)
    stack.append(summarizer)
    return stack


def drive(agent, config, history):  # noqa: ANN001
    for message in history:
        result = agent.invoke({"messages": [message]}, config=config)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            print("\n" + "=" * 68)
            print("memory-gate interrupted the run:")
            print("=" * 68)
            print(payload["text"])
            items = payload["items"]
            target = next(
                (i["ref"] for i in items if "fiscal year starts April" in i["preview"]), None
            )
            if target is None:
                answer = "confirm"
                print(f"\nyou -> {answer!r}   (nothing on this list is the constraint)")
            else:
                answer = f"keep {target}"
                print(f"\nyou -> {answer!r}   (pinning the fiscal-year constraint)")
            print("=" * 68)
            result = agent.invoke(Command(resume=answer), config=config)


def main() -> int:
    print("#" * 68)
    print("# PART A -- SummarizationMiddleware alone")
    print("#" * 68)
    observer_a = RecordWhatTheModelSees()
    agent_a = create_agent(
        model=GenericFakeChatModel(messages=iter([AIMessage(content="ok")] * 60)),
        tools=[],
        middleware=build_middlewares(observer_a),
        checkpointer=InMemorySaver(),
    )
    drive(agent_a, {"configurable": {"thread_id": "part-a"}}, HISTORY)
    probe = "fiscal year starts April"
    lost = probe not in observer_a.last_request_text
    print(f"\nmessages reaching the model: {observer_a.last_message_count}")
    print(f"system message:\n  {observer_a.last_system_message[:200]!r}")
    print(f"\n>>> {probe!r} present anywhere in the request: {probe in observer_a.last_request_text}")
    print(">>> nothing raised, nothing logged. The agent will now answer 'Q1' as Jan-Mar.")

    print("\n" + "#" * 68)
    print("# PART B -- MemoryGateMiddleware in front of it")
    print("#" * 68)
    workdir = Path(tempfile.mkdtemp(prefix="memory-gate-demo-"))
    try:
        observer_b = RecordWhatTheModelSees()
        gate = MemoryGateMiddleware(
            archive_dir=workdir,
            summarization=None,
            trigger=("messages", 10),
            keep_messages=4,
            review_threshold_tokens=1,
            protect_patterns=(r"fiscal year",),
        )
        # rebuild the summarizer so gate and agent share identical settings
        agent_b = create_agent(
            model=GenericFakeChatModel(messages=iter([AIMessage(content="ok")] * 60)),
            tools=[],
            middleware=build_middlewares(observer_b, gate),
            checkpointer=InMemorySaver(),
        )
        drive(agent_b, {"configurable": {"thread_id": "part-b"}}, HISTORY)
        kept = probe in observer_b.last_request_text
        # The point of this assertion: the original message really *was*
        # destroyed by compaction, and is back only because the gate put it in
        # the system message. Otherwise the demo would prove nothing.
        survived_naturally = probe in observer_b.last_body_text
        restored_by_gate = kept and not survived_naturally
        print(f"\nfinal system message reaching the model:\n  {observer_b.last_system_message[:400]!r}")
        print(f"\n>>> {probe!r} in the message window: {survived_naturally}  (compaction destroyed it)")
        print(f">>> present in the request via injection: {kept}")
        print(f">>> restored specifically by memory-gate: {restored_by_gate}")
        print(f"\narchive on disk: {gate.archive.stats()}")
        print(f"files in {workdir}:")
        for path in sorted(workdir.iterdir()):
            print(f"  {path.name:16s} {path.stat().st_size:>6} bytes")

        print("\npost-hoc recovery is available even for items you did not pin:")
        hits = gate.search("April")
        for record in hits[:3]:
            print(f"  #{record.id[:12]}  turn {record.turn:>2}  {record.text[:60]}")
        return 0 if (restored_by_gate and lost) else 1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

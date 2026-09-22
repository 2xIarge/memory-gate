"""Staged verification harness. Runs offline -- no API key, no network.

Each stage prints PASS/FAIL and the harness keeps going, so one failure does
not hide the rest. Output is written to a file because the Windows console
defaults to GBK and mangles the Chinese fixtures.

Stages
    1  package imports
    2  Archive is append-only, idempotent, and fail-closed on a bad path
    3  parse_reply fails closed on anything ambiguous
    4  before_model nodes run in middleware-list order  (claim #1)
    5  wrap_model_call composes first = outermost       (claim #2)
    6  interrupt() raised in before_model is NOT swallowed by
       ModelRetryMiddleware / ModelFallbackMiddleware    (claim #3)
    7  end-to-end: archive -> interrupt -> resume -> injection, and the
       lossy summary really does lose the pinned constraint without the gate
"""

from __future__ import annotations

import io
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

OUT = Path(__file__).with_name("verify_report.txt")
buf = io.StringIO()
RESULTS: list[tuple[str, bool, str]] = []


def say(*a):
    print(*a, file=buf)


def stage(n, title):
    say("")
    say("=" * 72)
    say(f"STAGE {n}  {title}")
    say("=" * 72)


def record(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    say(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def guard(name):
    """Decorator: capture exceptions so later stages still run."""
    def deco(fn):
        def wrapper(*a, **kw):
            try:
                return fn(*a, **kw)
            except Exception as exc:  # noqa: BLE001
                record(name, False, f"{type(exc).__name__}: {exc}")
                say(traceback.format_exc(limit=6))
                return None
        return wrapper
    return deco


# ---------------------------------------------------------------- stage 1
@guard("stage1 imports")
def stage1():
    stage(1, "package imports")
    import memory_gate  # noqa: F401
    from memory_gate import Archive, MemoryGateMiddleware, parse_reply, render_review_text  # noqa: F401
    say(f"  memory_gate {memory_gate.__version__} from {Path(memory_gate.__file__).parent}")
    record("imports", True)


# ---------------------------------------------------------------- stage 2
@guard("stage2 archive")
def stage2():
    stage(2, "Archive: append-only, idempotent, fail-closed")
    from memory_gate import Archive, ArchiveWriteError, Record

    tmp = Path(tempfile.mkdtemp(prefix="mg-arch-"))
    try:
        arch = Archive(tmp)
        recs = [
            Record(id=f"m{i}", role="human", turn=i, ts="t", tokens=10, text=f"msg {i}",
                   kind="dialogue" if i % 2 == 0 else "tool")
            for i in range(6)
        ]
        n1 = arch.append(recs)
        n2 = arch.append(recs)
        record("first append wrote 6", n1 == 6, f"got {n1}")
        record("re-append is idempotent", n2 == 0, f"got {n2}")
        record("seen_ids roundtrip", arch.seen_ids() == {r.id for r in recs})

        pinned = arch.protect([recs[0], recs[2]])
        record("protect wrote 2", pinned == 2, f"got {pinned}")
        record("protect is idempotent", arch.protect([recs[0]]) == 0)
        record("protected_ids", arch.protected_ids() == {"m0", "m2"})
        record("search finds text", [r.id for r in arch.search("msg 4")] == ["m4"])

        lines = (tmp / "archive.jsonl").read_text(encoding="utf-8").strip().split("\n")
        record("one JSON object per line", len(lines) == 6, f"got {len(lines)}")

        # fail-closed: an unwritable root must raise, not no-op
        bad = Archive(tmp / "a_file_not_a_dir" / "nested")
        (tmp / "a_file_not_a_dir").write_text("blocker", encoding="utf-8")
        try:
            bad.append(recs)
            record("unwritable archive raises", False, "no exception")
        except ArchiveWriteError as exc:
            record("unwritable archive raises", True, type(exc).__name__)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- stage 3
@guard("stage3 parse_reply")
def stage3():
    stage(3, "parse_reply fails closed")
    from memory_gate import parse_reply

    refs = {1, 2, 3, 4, 5}
    cases = [
        ("keep 1,3,5", "keep_selected", [1, 3, 5]),
        ("保留 2,4", "keep_selected", [2, 4]),
        ("3", "keep_selected", [3]),
        ("keep all", "keep_all", [1, 2, 3, 4, 5]),
        ("全部保留", "keep_all", [1, 2, 3, 4, 5]),
        ("confirm", "confirm", []),
        ("确认", "confirm", []),
        ("keep 百合花", "unrecognized", []),
        ("", "unrecognized", []),
        ("嗯这个我说不好", "unrecognized", []),
        ("keep 99", "unrecognized", []),
    ]
    for text, want_mode, want_refs in cases:
        got = parse_reply(text, refs)
        ok = got["mode"] == want_mode and got["keep_refs"] == want_refs
        record(f"parse {text!r}", ok, f"-> {got['mode']} {got['keep_refs']}")

    structured = parse_reply({"mode": "keep_selected", "keep_refs": [2]}, refs)
    record("structured dict reply", structured["mode"] == "keep_selected" and structured["keep_refs"] == [2])


# ---------------------------------------------------------------- stage 4-6
def _build_probe_agent(middlewares, model, checkpointer=None):
    from langchain.agents import create_agent
    return create_agent(model=model, tools=[], middleware=middlewares, checkpointer=checkpointer)


@guard("stage4 before_model order")
def stage4():
    stage(4, "before_model nodes run in middleware-list order")
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    order: list[str] = []

    def probe(label):
        # AgentMiddleware.name defaults to the class name and create_agent
        # rejects duplicate names, so every probe needs its own class.
        return type(
            f"Probe_{label}",
            (AgentMiddleware,),
            {"before_model": lambda self, state, runtime: (order.append(label), None)[1]},
        )()

    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")] * 4))
    agent = _build_probe_agent([probe("first"), probe("second"), probe("third")], model)
    agent.invoke({"messages": [("user", "hi")]})
    say(f"  observed order: {order}")
    record("list order == execution order", order[:3] == ["first", "second", "third"], str(order[:3]))


@guard("stage5 wrap_model_call outermost")
def stage5():
    stage(5, "wrap_model_call: first in list is outermost")
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    order: list[str] = []

    def wrapmw(label):
        def wrap(self, request, handler):
            order.append(f"enter:{label}")
            res = handler(request)
            order.append(f"exit:{label}")
            return res

        return type(f"Wrap_{label}", (AgentMiddleware,), {"wrap_model_call": wrap})()

    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")] * 4))
    agent = _build_probe_agent([wrapmw("outer"), wrapmw("inner")], model)
    agent.invoke({"messages": [("user", "hi")]})
    say(f"  observed: {order}")
    record(
        "first middleware wraps the second",
        order[:4] == ["enter:outer", "enter:inner", "exit:inner", "exit:outer"],
        str(order[:4]),
    )


@guard("stage6 interrupt survives retry+fallback")
def stage6():
    stage(6, "interrupt() in before_model is not swallowed by retry/fallback")
    from langchain.agents.middleware import (
        AgentMiddleware, ModelFallbackMiddleware, ModelRetryMiddleware,
    )
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    class Interrupter(AgentMiddleware):
        def before_model(self, state, runtime):
            from langgraph.types import interrupt
            answer = interrupt({"type": "probe", "q": "ping?"})
            say(f"  interrupt resumed with: {answer!r}")
            return None

    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")] * 6))
    # Interrupter is FIRST; retry/fallback sit inside it.
    # ModelFallbackMiddleware takes models positionally: (first_model, *additional).
    agent = _build_probe_agent(
        [Interrupter(), ModelRetryMiddleware(max_retries=2), ModelFallbackMiddleware(model)],
        model,
        checkpointer=InMemorySaver(),
    )
    cfg = {"configurable": {"thread_id": "probe-int"}}
    res = agent.invoke({"messages": [("user", "hi")]}, config=cfg)
    interrupted = "__interrupt__" in res
    say(f"  __interrupt__ present: {interrupted}")
    record("graph paused on interrupt", interrupted)

    if interrupted:
        res2 = agent.invoke(Command(resume="pong"), config=cfg)
        leaked = [m for m in res2.get("messages", []) if "ping" in str(getattr(m, "content", ""))]
        record("resume completed", "__interrupt__" not in res2)
        record("interrupt not converted to an AIMessage", not leaked, f"leaked={len(leaked)}")

    # And the adverse ordering: retry FIRST, interrupter inside it.
    order2: list[str] = []
    model2 = GenericFakeChatModel(messages=iter([AIMessage(content="ok")] * 6))

    class Interrupter2(AgentMiddleware):
        def before_model(self, state, runtime):
            from langgraph.types import interrupt
            interrupt({"type": "probe2"})
            return None

    agent2 = _build_probe_agent(
        [ModelRetryMiddleware(max_retries=2), Interrupter2()], model2, checkpointer=InMemorySaver()
    )
    cfg2 = {"configurable": {"thread_id": "probe-int-2"}}
    res3 = agent2.invoke({"messages": [("user", "hi")]}, config=cfg2)
    record(
        "before_model interrupt pauses even when retry is listed first",
        "__interrupt__" in res3,
        "before_model is its own graph node, outside the wrap chain",
    )


# ---------------------------------------------------------------- stage 7
FISCAL = "By the way, our fiscal year starts April 1st, so Q1 is Apr-Jun. All prior reports use that basis."
CANNED_SUMMARY = (
    "## SESSION INTENT\nAnalyse quarterly revenue.\n\n"
    "## SUMMARY\nUser asked for quarterly revenue analysis; several SQL queries were run.\n\n"
    "## ARTIFACTS\nNone\n\n## NEXT STEPS\nProduce the Q1 chart."
)


@guard("stage7 end-to-end")
def stage7():
    stage(7, "end-to-end: without the gate the constraint is lost, with it, it survives")
    from langchain.agents.middleware import AgentMiddleware, SummarizationMiddleware
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command
    from memory_gate import MemoryGateMiddleware

    class CaptureRequest(AgentMiddleware):
        """Records what the model actually receives, post-compaction.

        Absence must be asserted across the whole request, not just the system
        message -- in the no-gate case there is no system message at all, so
        checking only that would be trivially true and prove nothing.
        """
        def __init__(self):
            super().__init__()
            self.seen: list[dict] = []

        def wrap_model_call(self, request, handler):
            sm = getattr(request, "system_message", None)
            system = str(sm.content) if sm is not None else ""
            body = "\n".join(str(getattr(m, "content", "")) for m in request.messages)
            self.seen.append({
                "system": system,
                "body": body,
                "all": f"{system}\n{body}",
                "n_messages": len(request.messages),
            })
            return handler(request)

    def make_history(n_filler):
        msgs = [
            ("user", "Analyse our quarterly revenue for 2026."),
            ("assistant", "Sure. Which fiscal calendar should I use?"),
            ("user", FISCAL),
        ]
        for i in range(n_filler):
            msgs.append(("user", f"run query {i}"))
            msgs.append(("assistant", f"query {i} returned 100 rows, chart written to charts/q{i}.png"))
        msgs.append(("user", "Now give me the Q1 revenue chart."))
        return msgs

    trigger = ("messages", 10)
    keep = ("messages", 4)

    # ---- A: no gate -----------------------------------------------------
    say("\n  --- A) SummarizationMiddleware alone ---")
    cap_a = CaptureRequest()
    sum_model_a = GenericFakeChatModel(messages=iter([AIMessage(content=CANNED_SUMMARY)] * 3))
    model_a = GenericFakeChatModel(messages=iter([AIMessage(content="here is the chart")] * 30))
    summ_a = SummarizationMiddleware(model=sum_model_a, trigger=trigger, keep=keep)
    agent_a = _build_probe_agent([cap_a, summ_a], model_a, checkpointer=InMemorySaver())
    cfg_a = {"configurable": {"thread_id": "no-gate"}}
    for m in make_history(6):
        agent_a.invoke({"messages": [m]}, config=cfg_a)
    last_a = cap_a.seen[-1]
    probe = "fiscal year starts April"
    fiscal_in_a = probe in last_a["all"]
    say(f"  model requests observed: {len(cap_a.seen)}")
    say(f"  constraint anywhere in the final request: {fiscal_in_a}")
    say(f"  messages in final request: {last_a['n_messages']}")
    record("without the gate the constraint is GONE", not fiscal_in_a,
           "this is the bug the package exists to prevent")

    # ---- B: with gate, user pins it ------------------------------------
    say("\n  --- B) MemoryGate first in the list ---")
    tmp = Path(tempfile.mkdtemp(prefix="mg-e2e-"))
    try:
        cap_b = CaptureRequest()
        sum_model_b = GenericFakeChatModel(messages=iter([AIMessage(content=CANNED_SUMMARY)] * 3))
        model_b = GenericFakeChatModel(messages=iter([AIMessage(content="here is the chart")] * 30))
        summ_b = SummarizationMiddleware(model=sum_model_b, trigger=trigger, keep=keep)
        gate = MemoryGateMiddleware(
            archive_dir=tmp,
            summarization=summ_b,
            review_threshold_tokens=1,
            protect_patterns=(r"fiscal year",),
        )
        agent_b = _build_probe_agent([gate, cap_b, summ_b], model_b, checkpointer=InMemorySaver())
        cfg_b = {"configurable": {"thread_id": "with-gate"}}

        interrupts = []

        def drive(first_input):
            """Run to completion, answering every gate interrupt on the way."""
            res = agent_b.invoke(first_input, config=cfg_b)
            while "__interrupt__" in res:
                p = res["__interrupt__"][0].value
                interrupts.append(p)
                say(f"\n  >>> gate fired. review_id={p.get('review_id')} "
                    f"reason={p.get('reason')} items={len(p.get('items', []))} "
                    f"omitted={p.get('omitted')}")
                say("  >>> rendered for a plain-chat host:")
                for line in str(p.get("text", "")).splitlines():
                    say("      " + line)
                items = p.get("items", [])
                leaked = [i for i in items
                          if "Here is a summary of the conversation" in i.get("preview", "")]
                if leaked:
                    record("summary excluded from review list", False, f"{len(leaked)} leaked")
                target = next((i["ref"] for i in items
                               if "fiscal year starts April" in i.get("preview", "")), None)
                reply = f"keep {target}" if target is not None else "confirm"
                say(f"  >>> user replies: {reply!r}")
                res = agent_b.invoke(Command(resume=reply), config=cfg_b)
            return res

        for m in make_history(6):
            drive({"messages": [m]})

        record("gate interrupted at least once", len(interrupts) >= 1, f"{len(interrupts)} time(s)")
        # A gate that interrupts on nearly every turn gets uninstalled, which is
        # the failure mode the trigger alignment above exists to prevent.
        record("gate does not nag", len(interrupts) <= 4,
               f"{len(interrupts)} interrupt(s) across {len(make_history(6))} turns")
        st = gate.archive.stats()
        say(f"\n  archive stats: {st}")
        record("archive is non-empty", st["archived"] > 0, str(st))
        record("exactly one item pinned", st["protected"] == 1, str(st))

        last_b = cap_b.seen[-1]
        present_b = probe in last_b["all"]
        # If it were merely still in the window, the gate would be irrelevant.
        # It must be absent from the messages and present via the system block.
        natural_b = probe in last_b["body"]
        say(f"\n  constraint still in the message window: {natural_b}")
        say(f"  constraint present in final request:    {present_b}")
        say(f"\n  final system_message head:\n    {last_b['system'][:400]!r}")
        record("with the gate the constraint SURVIVES compaction", present_b)
        record("it survives via injection, not luck", present_b and not natural_b)

        runs = (tmp / "runs.jsonl").read_text(encoding="utf-8").strip().splitlines()
        record("every gate run left a trace", len(runs) >= 1, f"{len(runs)} run line(s)")
        say(f"  last run line: {runs[-1][:200]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@guard("stage8 guardrails")
def stage8():
    stage(8, "API guardrails: refuse silently-useless configurations")
    from langchain_core.messages import HumanMessage
    from memory_gate import Archive, GateError, MemoryGateMiddleware, Record
    from memory_gate.middleware import _kind_of

    tmp = Path(tempfile.mkdtemp(prefix="mg-guard-"))
    try:
        try:
            MemoryGateMiddleware(archive_dir=tmp / "a")
            record("missing trigger is refused", False, "constructed silently")
        except ValueError as exc:
            record("missing trigger is refused", True, str(exc)[:60])

        g = MemoryGateMiddleware(archive_dir=tmp / "b", trigger=("messages", 10))
        record("explicit trigger works standalone",
               g._compaction_is_imminent(10, 0) and not g._compaction_is_imminent(9, 0))

        g2 = MemoryGateMiddleware(
            archive_dir=tmp / "c", trigger=[{"tokens": 100, "messages": 5}, ("messages", 50)]
        )
        record("AND clause needs both",
               g2._compaction_is_imminent(5, 100) and not g2._compaction_is_imminent(5, 99))
        record("OR across clauses", g2._compaction_is_imminent(50, 0))

        summary = HumanMessage(
            content="Here is a summary of the conversation to date:\n\nx",
            additional_kwargs={"lc_source": "summarization"},
        )
        record("compaction summary excluded from review",
               _kind_of(summary) == "summary", _kind_of(summary))
        record("plain human stays reviewable",
               _kind_of(HumanMessage(content="keep this")) == "dialogue")

        arch = Archive(tmp / "d")
        arch.append([Record(id="x1", role="human", turn=1, ts="t", tokens=5,
                            text="fx rate rule", kind="dialogue")])
        g3 = MemoryGateMiddleware(archive_dir=tmp / "d", trigger=("messages", 2))
        rec = g3.restore("x1")
        record("restore pins an archived message", g3.archive.protected_ids() == {"x1"})
        record("restore returns the record", rec.text == "fx rate rule")
        try:
            g3.restore("nope")
            record("restore of unknown id raises", False, "no exception")
        except GateError:
            record("restore of unknown id raises", True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


for fn in (stage1, stage2, stage3, stage4, stage5, stage6, stage7, stage8):
    fn()

say("")
say("=" * 72)
passed = sum(1 for _, ok, _ in RESULTS if ok)
say(f"TOTAL {passed}/{len(RESULTS)} passed")
for name, ok, detail in RESULTS:
    if not ok:
        say(f"  FAILED: {name}  -- {detail}")
say("=" * 72)

OUT.write_text(buf.getvalue(), encoding="utf-8")
print(f"written -> {OUT}   ({passed}/{len(RESULTS)} passed)")

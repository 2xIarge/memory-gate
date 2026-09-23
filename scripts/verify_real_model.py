"""Adversarial real-model check: can compaction actually lose a prose constraint?

The first version of this check was too kind to the summariser. Every constraint
was announced as a rule ("记住几条：…"), all turns stayed on one topic, and
compaction ran exactly once -- so of course a 9B model kept them all. That run
proved nothing, in either direction.

This version is built to make loss *likely*:

  * constraints are slipped in incidentally, mid-request, never labelled
  * turns jump between five unrelated topics so there is no single narrative
    for the summary to hang things on
  * a low token trigger forces many compaction cycles, not one
  * one constraint is a pure arbitrary token (an internal codename) with no
    narrative role at all -- the classic casualty
  * every question is asked only at the very end, many compactions later

Guards that keep the measurement honest, learned the hard way:

  * a keyword may never appear inside its own question (that produced two
    fake PASSes before)
  * an empty reply, a non-assistant last message, or a verbatim echo of the
    question aborts instead of being scored
  * the run aborts unless compaction demonstrably fired more than once
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / ".scratch" / "real_model_report.txt"


@dataclass(frozen=True)
class Constraint:
    label: str
    plant: str        # the user turn that states it, in passing
    question: str     # asked only at the very end
    # AND-of-OR: every group must be matched, and a group matches when any
    # one of its alternatives appears. A bare "4" would let an unrelated
    # digit in the answer score as a correct fiscal quarter.
    must: tuple[tuple[str, ...], ...]


HARD = [
    Constraint(
        label="财年起始",
        plant="对了我们财年是从 4 月 1 号起的，你按 4 到 6 月当第一季度，帮我拉一下上季度营收。",
        question="我们的 Q1 覆盖哪几个月？只回答月份。",
        must=(("4", "四", "apr"), ("6", "六", "jun")),
    ),
    Constraint(
        label="营收口径",
        plant="营收那列用 net 打头的那个就行，gross 口径去年审计之后就不看了，帮我看看环比。",
        question="算营收该用哪个列？只回答列名。",
        must=(("net",),),
    ),
    Constraint(
        label="文件后缀",
        plant="导出的时候文件名结尾加个 _v3，我们内部都这么命名，习惯了。",
        question="导出文件名要加什么后缀？只回答后缀本身。",
        must=(("v3",),),
    ),
    Constraint(
        label="集中趋势",
        plant="集中趋势我们这边只用加权平均，中位数在这个场景没意义，按这个来。",
        question="统计集中趋势用哪种方法？只回答一种。",
        must=(("加权", "weighted"),),
    ),
    Constraint(
        label="对接人",
        plant="这个项目的对接人是 Lena，后面邮件直接发她，不用抄送我。",
        question="这个项目我发邮件给谁？只回答名字。",
        must=(("lena", "莉娜"),),
    ),
    Constraint(
        label="内部代号",
        plant="这单的内部代号是青橙-7，你顺手记一下，后面建目录要用。",
        question="这个项目的内部代号是什么？只回答代号。",
        must=(("青橙",),),
    ),
]

# Unrelated chatter. The point is that no single summary narrative can carry it.
CHATTER = [
    "帮我把这封英文邮件改得客气一点：'Your report is late again.'",
    "家里装修，75 寸电视和激光投影哪个性价比高？",
    "小孩暑假班是学编程好还是学围棋好？",
    "周末想带孩子去动物园，这个季节人多吗？",
    "帮我改一下这段 SQL：select * from orders limit 10;",
    "推荐三本讲谈判的书。",
    "明天会降温吗，要不要给孩子加件外套？",
    "咖啡和茶哪个提神更持久？",
    "团建去爬山还是去海边，十几个人。",
    "我家猫最近掉毛厉害，吃什么会好点？",
    "帮我算一下 3 月 15 号到 4 月 2 号一共多少天。",
    "Markdown 表格里怎么合并单元格？",
    "这个季度的差旅标准有变化吗？",
    "推荐一个免费的 PDF 阅读器。",
    "把上面那个结论写成三句话。",
    "阳台种什么植物不容易死？",
    "帮我想个会议室的名字，要有科技感。",
    "晚上十一点睡够八小时的话几点起？",
    "这段 Python 报 KeyError 一般是什么原因？",
    "给新人写一句欢迎语。",
]


def hard_turns() -> list[str]:
    """Interleave the planted constraints through the chatter."""
    plants = {i: c.plant for i, c in zip([0, 2, 4, 6, 9, 12], HARD)}
    turns: list[str] = []
    chatter = list(CHATTER)
    for i in range(len(CHATTER) + len(HARD)):
        if i in plants:
            turns.append(plants[i])
        else:
            turns.append(chatter.pop(0))
    return turns + ["就先这样，帮我把整体数字确认一遍。"]


def easy_turns() -> list[str]:
    return [
        "我在做 2026 年的季度营收复盘。",
        HARD[0].plant, HARD[1].plant, HARD[2].plant, HARD[3].plant, HARD[4].plant,
        "帮我把上个月的活跃用户数画个折线图。",
        "图例放到右边，字号大一点。",
        "把结果导出成 csv 给我。",
        "配色太亮了，换成公司蓝。",
        "再按部门拆一下。",
        "把标题改成英文。",
    ]


class MeasurementError(RuntimeError):
    """Anything that would make a score meaningless instead of a result."""


def validate_questions(constraints: list[Constraint]) -> None:
    """Refuse to run if a question contains its own answer keyword.

    That is not a hypothetical: an earlier version asked
    "用 net_revenue 还是 gross_revenue？" and scored a verbatim echo of the
    question as a correct answer.
    """
    for c in constraints:
        low = c.question.lower()
        for group in c.must:
          for kw in group:
            if kw.lower() in low:
                raise MeasurementError(
                    f"{c.label}: keyword {kw!r} appears inside its own question "
                    f"{c.question!r} -- any echo of the question would score PASS."
                )


def build(base_url: str, api_model: str, max_tokens: int, temperature: float = 0.0):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=api_model,
        base_url=base_url,
        api_key="not-needed",
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=300,
        # Qwen3.x defaults to a thinking mode that answers in
        # reasoning_content and leaves content empty.
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


def detect_model(base_url: str) -> str | None:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=5) as resp:
            items = (json.load(resp).get("data")) or []
        return items[0]["id"] if items else None
    except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError):
        return None


def ask(agent, config, question: str, resume_with: str) -> str:
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.types import Command

    result = agent.invoke({"messages": [HumanMessage(content=question)]}, config=config)
    while "__interrupt__" in result:
        result = agent.invoke(Command(resume=resume_with), config=config)
    messages = result.get("messages", [])
    if not messages:
        raise MeasurementError(f"empty state after asking {question!r}")
    last = messages[-1]
    if not isinstance(last, AIMessage):
        raise MeasurementError(
            f"no assistant reply for {question!r}; last message is {type(last).__name__}"
        )
    out = str(last.content)
    if not out.strip():
        raise MeasurementError(f"model returned empty content for {question!r}")
    if out.strip() == question.strip():
        raise MeasurementError(f"model echoed the question instead of answering: {question!r}")
    return out


def state_length(agent, config) -> int:
    state = agent.get_state(config)
    return len(state.values.get("messages", []) if state and state.values else [])


def state_tokens(agent, config) -> int:
    """Approximate token count, using the same estimator the summarizer uses."""
    from langchain_core.messages.utils import count_tokens_approximately

    state = agent.get_state(config)
    messages = state.values.get("messages", []) if state and state.values else []
    try:
        return int(count_tokens_approximately(messages))
    except Exception:  # noqa: BLE001 - estimator must never break the run
        return -1


def summarize_markers(agent, config) -> int:
    """Summary messages sitting in state *right now*.

    Not a compaction counter: with keep=4 the state always holds exactly one,
    no matter how many times compaction has fired.
    """
    return len(summary_ids(agent, config))


def summary_ids(agent, config) -> set[str]:
    """Ids of compaction summaries currently in state.

    ``SummarizationMiddleware._build_new_messages`` tags each summary with
    ``additional_kwargs={"lc_source": "summarization"}`` and never reuses an
    id, so the *distinct* ids observed across a run count compaction rounds.
    """
    state = agent.get_state(config)
    messages = state.values.get("messages", []) if state and state.values else []
    out: set[str] = set()
    for message in messages:
        if getattr(message, "additional_kwargs", {}).get("lc_source") != "summarization":
            continue
        mid = getattr(message, "id", None)
        if mid:
            out.add(str(mid))
        else:
            digest = hashlib.sha1(str(message.content).encode("utf-8")).hexdigest()[:12]
            out.add(f"unidentified:{digest}")
    return out


def run_turns(agent, config, turns, say, resume_with: str | None = None) -> int:
    """Feed the transcript, draining gate interrupts. Returns compaction count.

    Counting by "did the message list get shorter" was wrong and reported zero
    compactions while a summary marker sat in state: with keep=4 the list is
    pinned at 6 entries (summary + 4 kept + the incoming turn) forever, so it
    never shrinks. Distinct summary ids is the honest signal.
    """
    from langchain_core.messages import HumanMessage
    from langgraph.types import Command

    trace: list[tuple[int, int]] = []
    seen: set[str] = set()
    for text in turns:
        result = agent.invoke({"messages": [HumanMessage(content=text)]}, config=config)
        while "__interrupt__" in result:
            if resume_with is None:
                raise MeasurementError("unexpected interrupt in a gate-less run")
            result = agent.invoke(Command(resume=resume_with), config=config)
        new = summary_ids(agent, config) - seen
        if new:
            seen |= new
            say(f"    [compaction #{len(seen)}] new summary id(s): {sorted(new)}")
        trace.append((state_length(agent, config), state_tokens(agent, config)))
    say("  per-turn (messages, approx tokens):")
    say(f"    {trace}")
    return len(seen)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--scenario", choices=["hard", "easy"], default="hard")
    ap.add_argument("--trigger-tokens", type=int, default=320)
    ap.add_argument("--keep-messages", type=int, default=4)
    ap.add_argument("--parts", default="AB")
    args = ap.parse_args()

    buf = io.StringIO()

    def say(*a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True)
        print(line, file=buf)

    validate_questions(HARD)
    say("scoring self-check: no question contains its own keyword  [OK]")

    turns = hard_turns() if args.scenario == "hard" else easy_turns()
    say(f"scenario: {args.scenario}  ({len(turns)} user turns, {len(HARD)} planted constraints)")

    api_model = args.model or detect_model(args.base_url)
    if api_model is None:
        say(f"!! no OpenAI-compatible server at {args.base_url}")
        say("   start one:  pwsh -NoProfile -File scripts\\start_local_server.ps1")
        REPORT.write_text(buf.getvalue(), encoding="utf-8")
        return 2

    say(f"endpoint: {args.base_url}   model: {api_model}")

    from langchain.agents import create_agent
    from langchain.agents.middleware import SummarizationMiddleware
    from langgraph.checkpoint.memory import InMemorySaver
    from memory_gate import MemoryGateMiddleware

    agent_model = build(args.base_url, api_model, max_tokens=80)
    summary_model = build(args.base_url, api_model, max_tokens=500)
    say(f"connectivity: {str(agent_model.invoke('Reply with the single word: READY').content).strip()!r}")

    results: dict[str, dict[str, tuple[str, str]]] = {}
    meta: dict[str, dict[str, int]] = {}

    def make_summarizer():
        return SummarizationMiddleware(
            model=summary_model,
            trigger=("tokens", args.trigger_tokens),
            keep=("messages", args.keep_messages),
        )

    if "A" in args.parts.upper():
        say("\n" + "=" * 70)
        say("PART A -- SummarizationMiddleware alone")
        say("=" * 70)
        agent = create_agent(model=agent_model, tools=[], middleware=[make_summarizer()],
                             checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": f"{args.scenario}-a"}}
        n_compact = run_turns(agent, config, turns, say)
        say(f"  compaction cycles observed: {n_compact}")
        say(f"  summary markers in final state: {summarize_markers(agent, config)}")
        if n_compact < 2:
            raise MeasurementError(
                f"PART A: only {n_compact} compaction(s) ran. A single compaction is "
                "the mistake the first version made -- lower --trigger-tokens or add "
                "turns before reading anything into the score."
            )
        answers = {}
        for c in HARD:
            out = ask(agent, config, c.question, resume_with="confirm")
            ok = all(any(k.lower() in out.lower() for k in group)
                     for group in c.must)
            answers[c.label] = ("PASS" if ok else "FAIL", out)
            say(f"  [{c.label}] {'OK  ' if ok else 'LOST'}  {out[:90]!r}")
        results["A"] = answers
        meta["A"] = {"compactions": n_compact}

    if "B" in args.parts.upper():
        say("\n" + "=" * 70)
        say("PART B -- MemoryGateMiddleware in front of the same summarizer")
        say("=" * 70)
        archive_dir = ROOT / ".scratch" / f"gate-{args.scenario}"
        summarizer = make_summarizer()
        gate = MemoryGateMiddleware(
            archive_dir=archive_dir,
            summarization=summarizer,
            review_threshold_tokens=1,
            protect_patterns=tuple(c.must[0][0] for c in HARD)
            + ("只用", "不用", "按这个来", "记一下"),
        )
        agent = create_agent(model=agent_model, tools=[], middleware=[gate, summarizer],
                             checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": f"{args.scenario}-b"}}
        n_compact = run_turns(agent, config, turns, say, resume_with="keep all")
        say(f"  compaction cycles observed: {n_compact}")
        say(f"  archive: {gate.archive.stats()}")
        if n_compact < 2:
            raise MeasurementError(f"PART B: only {n_compact} compaction(s) ran.")
        answers = {}
        for c in HARD:
            out = ask(agent, config, c.question, resume_with="keep all")
            ok = all(any(k.lower() in out.lower() for k in group)
                     for group in c.must)
            answers[c.label] = ("PASS" if ok else "FAIL", out)
            say(f"  [{c.label}] {'OK  ' if ok else 'LOST'}  {out[:90]!r}")
        results["B"] = answers
        meta["B"] = {"compactions": n_compact}

    say("\n" + "=" * 70)
    say("RESULT")
    say("=" * 70)
    parts = list(results)
    say(f"{'constraint':<14}" + "".join(f"{p:>10}" for p in parts))
    for c in HARD:
        say(f"{c.label:<14}" + "".join(f"{results[p][c.label][0]:>10}" for p in parts))
    for p in parts:
        passed = sum(1 for v in results[p].values() if v[0] == "PASS")
        say(f"  part {p}: {passed}/{len(HARD)} survived   "
            f"({meta[p]['compactions']} compaction cycles)")

    lost_a = [c.label for c in HARD if results.get("A", {}).get(c.label, ("",))[0] == "FAIL"]
    if lost_a:
        say(f"\n>>> PART A lost {len(lost_a)} constraint(s) under compaction: {', '.join(lost_a)}")
        say(">>> that is the failure this package exists to make visible and recoverable.")
    else:
        say("\n>>> PART A lost nothing even on the adversarial transcript.")
        say(">>> The 'models drop constraints' premise is NOT supported by this run.")
        say(">>> Consider selling audit + recovery instead of loss-prevention.")

    REPORT.write_text(buf.getvalue(), encoding="utf-8")
    print(f"\nreport -> {REPORT}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MeasurementError as exc:
        message = f"MEASUREMENT ERROR: {exc}"
        print(f"\n\033[31m{message}\033[0m")
        try:
            previous = REPORT.read_text(encoding="utf-8")
        except OSError:
            previous = ""
        REPORT.write_text(previous + "\n" + message + "\n", encoding="utf-8")
        raise SystemExit(3)

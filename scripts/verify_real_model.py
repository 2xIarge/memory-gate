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
_RUN_TAG: str | None = None


def report_files() -> list[Path]:
    """Where to write: a per-configuration file first, then the stable "latest".

    Sharing one path across runs meant the second configuration of an evening
    overwrote the evidence from the first, and the only copy left said nothing
    about which policy produced it.
    """
    out = []
    if _RUN_TAG:
        out.append(REPORT.with_name(f"real_model_{_RUN_TAG}.txt"))
    out.append(REPORT)
    return out


def write_report(text: str) -> None:
    for path in report_files():
        path.write_text(text, encoding="utf-8")


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
    Constraint(
        label="数据环境",
        plant="查询走生产库那个只读副本，staging 的数据是脏的，谁也别往上写。",
        question="这次分析连的是哪套环境？只回答环境名。",
        must=(("生产", "prod"),),
    ),
    Constraint(
        label="货币单位",
        plant="金额一律换算成美元出报表，人民币的记得乘当季汇率。",
        question="报表里的金额用什么货币单位？",
        must=(("美元", "usd", "dollar"),),
    ),
    Constraint(
        label="交付节奏",
        plant="这个复盘每周五下班前必须交，拖到周一没人看。",
        question="这个复盘多久交一次、卡在星期几？",
        must=(("周五", "星期五", "friday"),),
    ),
    Constraint(
        label="标题规范",
        plant="图表标题记得带上季度号，不然是归档的时候根本检索不到。",
        question="图表标题需要额外带上什么信息？",
        must=(("季度",),),
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
    "帮我把这段中文摘要翻成英文，别太书面。",
    "会议室投影老断，是 HDMI 线的问题吗？",
    "推荐一个 macOS 上的窗口管理工具。",
    "这份报表发给老板之前要不要加一页结论？",
    "我们组有人要休产假，人手怎么排？",
    "帮我看看这段日志里有没有异常关键字。",
    "PPT 里的中文字体用哪个比较稳？",
    "季度目标怎么写才不至于太空？",
    "楼下那家咖啡店涨价了，附近有平替吗？",
    "帮我把这个 CSV 的前五行打出来看看。",
    "这个字段命名是 snake_case 还是 camelCase？",
    "团队周会要不要改成双周一次？",
    "打印机又卡纸了，换个纸盒能解决吗？",
    "把这段 SQL 改成参数化的，别拼字符串。",
    "有没有适合新人的代码评审清单？",
    "这个报错一般是什么权限问题？",
    "帮我想个不那么俗的项目代号。",
    "年底团建预算人均多少合适？",
    "这份文档放 wiki 哪个目录比较好？",
    "键盘手感差，换轴能救吗？",
    "把上面几条结论合并成一段话。",
    "客户问交付延期怎么解释比较得体？",
    "这个表数据量多大，需要分页吗？",
    "午饭吃啥，别太油。",
    "帮我把这段正则解释一下。",
    "我们的日志保留多久比较合规？",
]


def hard_turns() -> list[str]:
    """Interleave the planted constraints evenly through the chatter.

    Spacing matters: a constraint planted right before a compaction boundary is
    easy to keep, one planted many compactions ago is the hard case. Even
    spacing samples both.
    """
    total = len(CHATTER) + len(HARD)
    step = total // len(HARD)
    positions = {min(total - 1, i * step + 1): c.plant for i, c in enumerate(HARD)}
    turns: list[str] = []
    chatter = list(CHATTER)
    for i in range(total):
        if i in positions:
            turns.append(positions[i])
        elif chatter:
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
        if not isinstance(c.must, tuple) or not c.must:
            raise MeasurementError(f"{c.label}: `must` must be a non-empty tuple of groups")
        for gi, group in enumerate(c.must):
            if not isinstance(group, tuple) or not group                     or not all(isinstance(k, str) and k for k in group):
                raise MeasurementError(
                    f"{c.label}: group #{gi} is {group!r}; every group must be a non-empty "
                    "tuple of non-empty strings. A single group needs a trailing comma -- "
                    '((a, b)) collapses to (a, b) and then scores on single characters.'
                )
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
        # resolve_resume, not the raw value: `resume_with` may be a policy
        # callable, and Command(resume=<function>) makes LangGraph try to
        # checkpoint a function and die in msgpack.
        payload = result["__interrupt__"][0].value
        result = agent.invoke(
            Command(resume=resolve_resume(resume_with, payload)), config=config
        )
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


def resolve_resume(resume_with, payload) -> str:
    """`resume_with` is either a fixed reply or a policy callable."""
    return resume_with(payload) if callable(resume_with) else resume_with


def policy_flagged(payload) -> str:
    """A realistic reviewer: pin only what the gate flagged, skip the rest."""
    # module-level function, so it cannot see main()'s imports. Same trap as
    # ask(): anything a top-level helper needs must be imported where it lives.
    from memory_gate import flagged_refs

    refs = flagged_refs(payload)
    return "keep " + ",".join(str(r) for r in refs) if refs else "confirm"


def run_turns(agent, config, turns, say, resume_with: str | None = None) -> tuple[int, int]:
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
    reviews = 0
    for text in turns:
        result = agent.invoke({"messages": [HumanMessage(content=text)]}, config=config)
        while "__interrupt__" in result:
            if resume_with is None:
                raise MeasurementError("unexpected interrupt in a gate-less run")
            reviews += 1
            payload = result["__interrupt__"][0].value
            # Capture what a human would actually see. Statistics alone hid the
            # fact that no run had ever recorded a review block.
            say("")
            say(f"  ===== REVIEW {reviews} (after turn {len(trace) + 1}) =====")
            for line in str(payload.get("text", "")).splitlines():
                say("  " + line)
            answer = resolve_resume(resume_with, payload)
            say(f"  you -> {answer!r}")
            say("  ========================================")
            result = agent.invoke(Command(resume=answer), config=config)
        new = summary_ids(agent, config) - seen
        if new:
            seen |= new
            say(f"    [compaction #{len(seen)}] new summary id(s): {sorted(new)}")
        trace.append((state_length(agent, config), state_tokens(agent, config)))
    say("  per-turn (messages, approx tokens):")
    say(f"    {trace}")
    return len(seen), reviews


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default=None)
    ap.add_argument("--scenario", choices=["hard", "easy"], default="hard")
    ap.add_argument("--trigger-tokens", type=int, default=320)
    ap.add_argument("--keep-messages", type=int, default=4)
    ap.add_argument("--parts", default="AB")
    ap.add_argument("--protect-budget", type=int, default=None,
                    help="cap on pinned tokens injected per model call; omit to use "
                         "the library default (2% of the trigger)")
    ap.add_argument("--policy", choices=["all", "flagged"], default="all",
                    help="how part B answers the gate: pin everything, or only flagged")
    ap.add_argument("--patterns", choices=["oracle", "default"], default="default",
                    help="oracle = protect_patterns built from the planted constraints' "
                         "own keywords, i.e. the reviewer holds the answer key. That is "
                         "what the first real-model run did, and it cannot say anything "
                         "about triage. default = the shipped DEFAULT_PROTECT_PATTERNS.")
    args = ap.parse_args()

    # One tag for one configuration, used for the archive directory, the graph
    # thread and the report file. Without it two runs in one evening shared an
    # archive, so the second inherited the first's keep.jsonl and both sets of
    # stats were cumulative.
    global _RUN_TAG
    _RUN_TAG = f"{args.scenario}-{args.policy}-{args.patterns}" + (
        "" if args.protect_budget is None else f"-b{args.protect_budget}")

    buf = io.StringIO()

    def say(*a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True)
        print(line, file=buf)

    validate_questions(HARD)
    say("scoring self-check: no question contains its own keyword  [OK]")

    turns = hard_turns() if args.scenario == "hard" else easy_turns()
    say(f"scenario: {args.scenario}  ({len(turns)} user turns, {len(HARD)} planted constraints)")
    say(f"run tag : {_RUN_TAG}")
    say(f"policy  : {args.policy}   trigger: ({'tokens'}, {args.trigger_tokens}) "
        f"keep: ({'messages'}, {args.keep_messages})")
    if args.protect_budget is None:
        say("budget  : library default (2% of the trigger, clamped to [200, 4000])")
    else:
        say(f"budget  : {args.protect_budget} tokens injected per call = "
            f"{args.protect_budget / args.trigger_tokens:.0%} of the trigger")
    if args.patterns == "oracle":
        say("triage  : **ORACLE** -- protect_patterns are the planted constraints' own "
            "keywords.\n          This run can show that pinning survives compaction. It "
            "cannot show that the\n          flag set finds the right lines; do not "
            "quote it as if it could.")
    else:
        say("triage  : shipped DEFAULT_PROTECT_PATTERNS -- the reviewer gets no keyword "
            "advantage\n          against the planted constraints.")

    api_model = args.model or detect_model(args.base_url)
    if api_model is None:
        say(f"!! no OpenAI-compatible server at {args.base_url}")
        say("   start one:  pwsh -NoProfile -File scripts\\start_local_server.ps1")
        write_report(buf.getvalue())
        return 2

    say(f"endpoint: {args.base_url}   model: {api_model}")

    from langchain.agents import create_agent
    from langchain.agents.middleware import SummarizationMiddleware
    from langgraph.checkpoint.memory import InMemorySaver
    from memory_gate import DEFAULT_PROTECT_PATTERNS, MemoryGateMiddleware

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
        config = {"configurable": {"thread_id": f"{_RUN_TAG}-a"}}
        n_compact, n_review = run_turns(agent, config, turns, say)
        say(f"  compaction cycles observed: {n_compact}   gate reviews: {n_review}")
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
        meta["A"] = {"compactions": n_compact, "reviews": n_review}

    if "B" in args.parts.upper():
        say("\n" + "=" * 70)
        say("PART B -- MemoryGateMiddleware in front of the same summarizer")
        say("=" * 70)
        archive_dir = ROOT / ".scratch" / f"gate-{_RUN_TAG}"
        summarizer = make_summarizer()
        planted_keys = tuple(c.must[0][0] for c in HARD)
        if args.patterns == "oracle":
            chosen = planted_keys + ("只用", "不用", "按这个来", "记一下")
        else:
            chosen = tuple(DEFAULT_PROTECT_PATTERNS)
            # The whole point of --patterns default is that the reviewer does not
            # know the answers. Report any accidental overlap instead of assuming
            # there is none: `口径` ships in the default set and is also a word the
            # revenue constraint uses, and a reader of this report is entitled to
            # know that the comparison is not perfectly clean.
            leak = [k for k in planted_keys
                    if any(k.lower() in p.lower() or p.lower() in k.lower()
                           for p in chosen)]
            say(f"  oracle leak check: planted answer keywords also present in the "
                f"default set -> {leak or 'none'}")
        say(f"  patterns: {args.patterns} ({len(chosen)} entries)")
        gate = MemoryGateMiddleware(
            archive_dir=archive_dir,
            summarization=summarizer,
            review_threshold_tokens=1,
            protect_budget_tokens=args.protect_budget,
            protect_patterns=chosen,
        )
        agent = create_agent(model=agent_model, tools=[], middleware=[gate, summarizer],
                             checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": f"{_RUN_TAG}-b"}}
        policy = policy_flagged if args.policy == "flagged" else "keep all"
        n_compact, n_review = run_turns(agent, config, turns, say, resume_with=policy)
        say(f"  compaction cycles observed: {n_compact}   gate reviews: {n_review}")
        say(f"  gate interrupted {n_review} time(s)")
        say(f"  archive: {gate.archive.stats()}")
        if n_compact < 2:
            raise MeasurementError(f"PART B: only {n_compact} compaction(s) ran.")
        answers = {}
        for c in HARD:
            out = ask(agent, config, c.question,
                      resume_with=policy_flagged if args.policy == "flagged" else "keep all")
            ok = all(any(k.lower() in out.lower() for k in group)
                     for group in c.must)
            answers[c.label] = ("PASS" if ok else "FAIL", out)
            say(f"  [{c.label}] {'OK  ' if ok else 'LOST'}  {out[:90]!r}")
        results["B"] = answers
        meta["B"] = {"compactions": n_compact, "reviews": n_review}

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
            f"({meta[p]['compactions']} compaction cycles, "
            f"{meta[p]['reviews']} gate review(s))")

    lost_a = [c.label for c in HARD if results.get("A", {}).get(c.label, ("",))[0] == "FAIL"]
    if lost_a:
        say(f"\n>>> PART A lost {len(lost_a)} constraint(s) under compaction: {', '.join(lost_a)}")
        say(">>> that is the failure this package exists to make visible and recoverable.")
    else:
        say("\n>>> PART A lost nothing even on the adversarial transcript.")
        say(">>> The 'models drop constraints' premise is NOT supported by this run.")
        say(">>> Consider selling audit + recovery instead of loss-prevention.")

    write_report(buf.getvalue())
    print(f"\nreport -> {report_files()[0]}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MeasurementError as exc:
        message = f"MEASUREMENT ERROR: {exc}"
        print(f"\n\033[31m{message}\033[0m")
        target = report_files()[0]
        try:
            previous = target.read_text(encoding="utf-8")
        except OSError:
            previous = ""
        # An aborted run still has to leave a file behind, and that file has to
        # say it aborted -- a partial transcript that reads like a finished one is
        # how "8/10 survived" got written down once already.
        write_report(previous + "\n" + message + "\n")
        raise SystemExit(3)

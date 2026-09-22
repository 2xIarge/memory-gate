"""Real-model check for memory-gate.

Everything in ``scripts/verify.py`` runs on ``GenericFakeChatModel``, which
proves the middleware integrates with LangChain's compaction machinery but says
nothing about what an actual LLM chooses to throw away. This script asks that
question: it drives a real model through a conversation containing several
planted constraints, lets ``SummarizationMiddleware`` compact, then queries
each constraint again.

Start any OpenAI-compatible server first, e.g.

    pwsh -NoProfile -File scripts\\start_local_server.ps1

then

    .\\.venv\\Scripts\\python.exe scripts\\verify_real_model.py

Two parts, two fresh threads, same conversation transcript:

  A  agent + SummarizationMiddleware          <- what the host does today
  B  agent + MemoryGateMiddleware + same      <- gate pins every constraint

Score = how many planted constraints the model can still state correctly after
compaction.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / ".scratch" / "real_model_report.txt"

# (question, keywords any-correct-answer must contain, short label)
CONSTRAINTS = [
    ("我们财年从几月开始？Q1 对应哪几个月份？只回答月份。", ("4", "四", "apr", " april"), "财年起始"),
    ("算营收应该用 net_revenue 还是 gross_revenue？只回答列名。", ("net",), "营收口径"),
    ("导出文件名要加什么后缀？只回答后缀。", ("v3",), "文件后缀"),
    ("统计集中趋势用中位数还是加权平均？只回答一种。", ("加权", "weighted"), "集中趋势"),
    ("这次项目的对接人叫什么？只回答名字。", ("lena", "莉娜"), "对接人"),
]

# Deliberately unrelated chatter: it is what pushes the constraints out of the
# keep window, and it is plausible for a real session.
FILLER = [
    "帮我把上个月的活跃用户数画个折线图。",
    "图例放到右边，字号大一点。",
    "这个查询太慢了，加个索引建议。",
    "把结果导出成 csv 给我。",
    "顺便看看有没有空值。",
    "配色太亮了，换成公司蓝。",
    "再按部门拆一下。",
    "这个异常值帮我标出来。",
    "把标题改成英文。",
    "导出目录建在哪了？",
]

OPENING = [
    "我在做 2026 年的季度营收复盘，接下来所有分析都按我们的惯例来。",
    "记住几条：财年从 4 月 1 号开始，Q1 是 4 到 6 月。",
    "营收一律用 net_revenue，不要用 gross_revenue，去年审计之后改的。",
    "导出的文件名结尾都要加 _v3。",
    "集中趋势不要用中位数，只用加权平均。",
    "这个项目的对接人是 Lena，邮件直接发她。",
]


def build(base_url: str, api_model: str, max_tokens: int, temperature: float = 0.0):
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=api_model,
        base_url=base_url,
        api_key="not-needed",
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=300,
    )
    return llm


def detect_model(base_url: str) -> str | None:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=5) as resp:
            data = json.load(resp)
        items = data.get("data") or []
        return items[0]["id"] if items else None
    except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError):
        return None


def probe(llm) -> str:
    return str(llm.invoke("say OK").content)[:80]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default=None, help="API model name; auto-detected if omitted")
    ap.add_argument("--trigger-tokens", type=int, default=1200)
    ap.add_argument("--keep-messages", type=int, default=4)
    ap.add_argument("--parts", default="AB", help="subset of A,B")
    args = ap.parse_args()

    buf = io.StringIO()

    def say(*a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True)
        print(line, file=buf)

    api_model = args.model or detect_model(args.base_url)
    if api_model is None:
        say(f"!! no OpenAI-compatible server answering at {args.base_url}")
        say("   start one first:  pwsh -NoProfile -File scripts\\start_local_server.ps1")
        REPORT.write_text(buf.getvalue(), encoding="utf-8")
        return 2

    say(f"endpoint: {args.base_url}")
    say(f"model   : {api_model}")

    from langchain.agents import create_agent
    from langchain.agents.middleware import SummarizationMiddleware
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command
    from memory_gate import MemoryGateMiddleware

    agent_model = build(args.base_url, api_model, max_tokens=70)
    summary_model = build(args.base_url, api_model, max_tokens=450, temperature=0.0)
    say(f"connectivity: {probe(agent_model)!r}")

    transcript = [(m, "user") for m in OPENING + FILLER]
    # interleave a short assistant turn for each user turn to look like a real chat
    conversation: list[tuple[str, str]] = []
    for i, (msg, _) in enumerate(transcript):
        conversation.append((msg, "user"))
        conversation.append((f"收到，第{i}步已处理。", "assistant"))
    conversation.append(("我刚才提的那几条规矩都记住了吧？", "user"))
    conversation.append(("记住了。", "assistant"))

    def make_summarizer() -> SummarizationMiddleware:
        return SummarizationMiddleware(
            model=summary_model,
            trigger=("tokens", args.trigger_tokens),
            keep=("messages", args.keep_messages),
        )

    results: dict[str, dict[str, str]] = {}

    if "A" in args.parts.upper():
        say("\n" + "=" * 68)
        say("PART A -- SummarizationMiddleware alone (what a host does today)")
        say("=" * 68)
        summarizer = make_summarizer()
        agent = create_agent(model=agent_model, tools=[], middleware=[summarizer],
                             checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "real-a"}}
        for text, role in conversation:
            from langchain_core.messages import AIMessage, HumanMessage
            msg = HumanMessage(content=text) if role == "user" else AIMessage(content=text)
            agent.invoke({"messages": [msg]}, config=config)
        answers = {}
        for question, keywords, label in CONSTRAINTS:
            out = str(agent.invoke({"messages": [("user", question)]}, config=config)
                      ["messages"][-1].content)
            ok = any(k.lower() in out.lower() for k in keywords)
            answers[label] = ("PASS" if ok else "FAIL", out[:120])
            say(f"  [{label}] {'OK ' if ok else 'LOST'}  {out[:110]!r}")
        results["A"] = answers

    if "B" in args.parts.upper():
        say("\n" + "=" * 68)
        say("PART B -- MemoryGateMiddleware in front of the same summarizer")
        say("=" * 68)
        archive_dir = ROOT / ".scratch" / "real-gate"
        summarizer = make_summarizer()
        gate = MemoryGateMiddleware(
            archive_dir=archive_dir,
            summarization=summarizer,
            review_threshold_tokens=1,
            protect_patterns=("财年", "net_revenue", "_v3", "加权", "中位数", "对接人", "记住"),
        )
        agent = create_agent(model=agent_model, tools=[],
                             middleware=[gate, summarizer],
                             checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "real-b"}}
        asked = 0
        for text, role in conversation:
            from langchain_core.messages import AIMessage, HumanMessage
            msg = HumanMessage(content=text) if role == "user" else AIMessage(content=text)
            result = agent.invoke({"messages": [msg]}, config=config)
            while "__interrupt__" in result:
                payload = result["__interrupt__"][0].value
                asked += 1
                if asked <= 3:
                    say("  --- gate asked ---")
                    for line in payload["text"].splitlines():
                        say("  " + line)
                # pin everything: this is the "user cares about all of it" case
                result = agent.invoke(Command(resume="keep all"), config=config)
        say(f"  gate interrupted {asked} time(s)")
        answers = {}
        for question, keywords, label in CONSTRAINTS:
            out = str(agent.invoke({"messages": [("user", question)]}, config=config)
                      ["messages"][-1].content)
            ok = any(k.lower() in out.lower() for k in keywords)
            answers[label] = ("PASS" if ok else "FAIL", out[:120])
            say(f"  [{label}] {'OK ' if ok else 'LOST'}  {out[:110]!r}")
        results["B"] = answers
        say(f"  archive: {gate.archive.stats()}")

    say("\n" + "=" * 68)
    say("SUMMARY   (PASS = constraint still stated correctly after compaction)")
    say("=" * 68)
    labels = [c[2] for c in CONSTRAINTS]
    header = f"{'constraint':<14}" + "".join(f"{p:>10}" for p in results)
    say(header)
    for label in labels:
        row = f"{label:<14}"
        for part in results:
            verdict = results[part].get(label, ("-", ""))[0]
            row += f"{verdict:>10}"
        say(row)
    for part in results:
        passed = sum(1 for v in results[part].values() if v[0] == "PASS")
        say(f"  part {part}: {passed}/{len(labels)} constraints survived")
    say("\nread this honestly: if A kept everything, the real model simply summarised")
    say("well on this transcript and this run proves nothing about loss.")

    REPORT.write_text(buf.getvalue(), encoding="utf-8")
    print(f"\nreport -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

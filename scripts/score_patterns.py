"""Score ``DEFAULT_PROTECT_PATTERNS`` and report what they actually catch.

    python scripts/score_patterns.py

The flags are the only triage signal a reviewer gets, so their hit rate is a
product property, not an implementation detail. This script exists because the
first version of the pattern list was tuned by reading the same transcript it
was then scored on, and reported recall that did not survive contact with new
sentences.

Three corpora, separated on purpose:

  planted   the 10 constraints planted in a recorded 57-turn run. The
            declarative markers were derived by reading these, so this column is
            a training score and must never be quoted as accuracy.
  heldout1  14 constraints + 14 chatter lines. The shipped candidate was
            revised after reading its misses, so it is a training set too.
  heldout2  12 constraints + 12 chatter lines in different domains and
            vocabulary. Written after the pattern list was fixed and never
            iterated against. This is the number to quote, with the caveat that
            author and scorer were the same session.

The recorded-run column needs ``.scratch/gate-hard/archive.jsonl``, which is
gitignored; without it the script still scores the built-in corpora.

Also kept here: ``V2``, the revision that looked better and was not. It lifted
planted recall from 4/10 to 8/10 while *lowering* held-out recall, because it
was written against the planted list and dropped the bare ``别`` pattern that
real directives such as "别用 seaborn" depend on. Deleting it would erase the
reason the corpora are split.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from memory_gate.middleware import DEFAULT_PROTECT_PATTERNS  # noqa: E402

FLAGS = re.IGNORECASE          # must match middleware.py, which compiles with it
ARCHIVE = ROOT / ".scratch/gate-hard/archive.jsonl"

SHIPPED = tuple(DEFAULT_PROTECT_PATTERNS)

# Frozen literally, never derived from SHIPPED. Deriving them by subtraction
# silently redefines the control group every time the defaults are edited: the
# first version of this file computed V2 as SHIPPED minus two patterns, and
# after those two patterns were replaced the two columns scored identically --
# a comparison that had stopped comparing anything.
ORIGINAL = (
    r"\bmust\b", r"\bnever\b", r"\balways\b", r"\bdo not\b", r"\bdon't\b",
    r"\bforbidden\b", r"\brequire[sd]?\b", r"\bconstraint\b", r"\bpolicy\b",
    r"\bcomply\b", r"\bcompliance\b", r"\bonly use\b", r"\bremember\b",
    r"必须", r"不允许", r"不得", r"禁止", r"一律", r"务必", r"口径",
    r"约定", r"记住", r"千万", r"只能", r"不要", r"别",
)

V2 = (
    r"\bmust\b", r"\bnever\b", r"\balways\b", r"\bdo not\b", r"\bdon't\b",
    r"\bforbidden\b", r"\brequire[sd]?\b", r"\bconstraint\b", r"\bpolicy\b",
    r"\bcomply\b", r"\bcompliance\b", r"\bonly use\b", r"\bremember\b",
    r"必须", r"不允许", r"不得", r"禁止", r"一律", r"务必", r"口径",
    r"约定", r"记住", r"千万", r"只能",
    r"(?<!要)不要",
    r"千万别", r"别忘", r"谁都别", r"谁也别",
    r"记得", r"记一下", r"顺手记", r"记下来",
    r"都这么", r"按这个来", r"习惯了", r"统一", r"一贯", r"向来", r"一直是用",
    r"(我们|咱们|这边|内部|公司|团队|部门|组里)(这边|内部)?(是|用|只|都|统一|按|叫|走|默认)",
    r"\bwe (use|always|never|only|call|prefer|default to)\b",
    r"\bour \w+ (is|are|starts?|runs?|lives?)\b",
    r"\bby convention\b", r"\bfor future reference\b", r"\bnote that\b",
    r"\bgoing forward\b", r"\bfrom now on\b", r"\bis called\b",
    r"\bwe refer to\b", r"\bheads[- ]up\b", r"\bFYI\b",
)

PLANTED = {
    "财年起始": "财年是从 4 月 1 号起", "营收口径": "net 打头的那个",
    "文件后缀": "_v3", "集中趋势": "只用加权平均", "对接人": "对接人是 Lena",
    "内部代号": "青橙-7", "数据环境": "只读副本", "货币单位": "换算成美元",
    "交付节奏": "每周五下班前", "标题规范": "季度号",
}

HELDOUT1_CONSTRAINTS = (
    "我们代码库统一用 4 空格缩进。", "数据库密码放在 vault 里，不要写进配置文件。",
    "周报发给 Kevin，抄送整个组。", "版本号按 SemVer 走。",
    "所有图表都用 matplotlib 出，别用 seaborn。", "客户那边只认 PDF，word 他们打不开。",
    "我们的时区统一按 UTC+8 记。", "这个服务的负责人是小赵，出问题先找他。",
    "Deploy branch is main, never push directly to prod.",
    "We use UTC for all timestamps in the logs.",
    "Our fiscal year starts in April.",
    "By convention the export suffix is _final.",
    "Going forward all invoices must be in EUR.",
    "The staging database is read-only, nobody writes to it.",
)
HELDOUT1_CHATTER = (
    "今天午饭吃啥？", "明天会下雨吗？", "帮我写一首关于秋天的诗。", "这个函数是干嘛的？",
    "要不要一起去看电影？", "推荐几本科幻小说。", "我家狗不吃狗粮怎么办？",
    "帮我把这段话缩短一点。", "What is the weather like tomorrow?",
    "Can you summarize this article for me?", "Should we get pizza or noodles?",
    "Tell me a joke about programmers.", "这段代码跑得有点慢，怎么回事？",
    "帮我改个错别字。",
)

HELDOUT2_CONSTRAINTS = (
    "所有对外数据都得过一遍合规审查再发。", "报表里的金额保留两位小数，四舍五入。",
    "我们的主分支叫 release，不叫 main。", "客户名单不许导出成 Excel。",
    "每周三下午三点站会，别迟到。", "日志级别生产环境固定为 warn。",
    "发票开给上海主体，不要开给深圳。", "API key 只能放在环境变量里。",
    "All customer emails go out from support@, never from a personal address.",
    "Our SLA is four business hours, not four hours.",
    "Screenshots in the docs must be taken at 1440p.",
    "The primary key is always a UUID, never an incrementing integer.",
)
HELDOUT2_CHATTER = (
    "你觉得这个配色好看吗？", "帮我起个变量名。", "今晚吃什么好？", "这首歌叫什么名字？",
    "要不要试试换个写法？", "我有点困，讲个笑话吧。", "附近有什么好吃的日料？",
    "帮我把这段英文读一遍。", "Which movie should I watch tonight?",
    "Can you explain what a closure is?", "Do you think it will rain this weekend?",
    "我买的新手机屏碎了，能修吗？",
)


def compile_set(patterns):
    return [re.compile(p, FLAGS) for p in patterns]


def hit(patterns, text):
    return any(p.search(text) for p in patterns)


def recorded_user_messages():
    """The user turns from a recorded run, or [] when the archive is absent."""
    if not ARCHIVE.exists():
        return []
    rows = [json.loads(l) for l in ARCHIVE.read_text(encoding="utf-8").splitlines()]
    return [r["text"] for r in rows
            if r["kind"] == "dialogue" and r["role"] == "human"]


def report(label, patterns, recorded):
    print(f"\n=== {label} ===")
    if recorded:
        texts = {n: next((t for t in recorded if f in t), None)
                 for n, f in PLANTED.items()}
        missing = [n for n, t in texts.items() if t is None]
        if missing:
            print(f"  planted  skipped, transcript lacks: {', '.join(missing)}")
        else:
            caught = [n for n, t in texts.items() if hit(patterns, t)]
            print(f"  planted  (training) {len(caught)}/{len(texts)}   missed:"
                  f" {', '.join(n for n in texts if n not in caught) or '-'}")
    for tag, cons, chat in (("heldout1", HELDOUT1_CONSTRAINTS, HELDOUT1_CHATTER),
                            ("heldout2", HELDOUT2_CONSTRAINTS, HELDOUT2_CHATTER)):
        tp = [c for c in cons if hit(patterns, c)]
        fn = [c for c in cons if not hit(patterns, c)]
        fp = [c for c in chat if hit(patterns, c)]
        print(f"  {tag}  recall {len(tp)}/{len(cons)} = {len(tp)/len(cons):.0%}"
              f"   precision {len(tp)}/{len(tp)+len(fp)}"
              f" = {len(tp)/max(len(tp)+len(fp),1):.0%}")
        for c in fn:
            print(f"       MISS {c}")
        for c in fp:
            print(f"       FP   {c}")
    if recorded:
        flagged = sum(1 for t in recorded if hit(patterns, t))
        print(f"  recorded flag rate {flagged}/{len(recorded)}"
              f" = {flagged/len(recorded):.0%}"
              "   (how much of a real transcript the reviewer must triage)")


def main() -> int:
    recorded = recorded_user_messages()
    print(f"compiled with IGNORECASE, as middleware.py does")
    print(f"recorded user messages: {len(recorded)}"
          + ("" if recorded else "  (archive absent, scoring built-in corpora only)"))
    for label, patterns in (("ORIGINAL (imperative only)", ORIGINAL),
                            ("V2 (the overfit one)", V2),
                            ("SHIPPED (current defaults)", SHIPPED)):
        report(label, compile_set(patterns), recorded)
    print("\nA pattern list has a ceiling: a rule stated with no marker word at"
          "\nall -- 周报发给 Kevin，抄送整个组 -- is unreachable. That is why the"
          "\nreview's default action keeps everything rather than trusting these.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

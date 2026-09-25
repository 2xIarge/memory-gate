"""What does the review list actually look like at production scale?

The gate has only ever been eyeballed at a 2 500-token trigger, because that is
what fits a 52 tok/s local model in a few minutes. Nobody runs a window that
small: the case this package has to serve is ~200K with compaction firing near
80%, i.e. ~160K tokens leaving the window at once.

This script renders that. It calls no model and changes no library code -- it
feeds measured inputs through the real ``render_review_text``, so the output is
the product's own formatting rather than a paraphrase of it.

Measured inputs come from .scratch/gate-hard/archive.jsonl, the real 57-turn
run: the mean human turn is ~9 tokens, which makes the list length equal to the
number of user turns, and the user turns themselves are replayed as content so
the preview text is verbatim real speech instead of invented filler. How many
user turns fill 160K depends on how much else each turn drags along, so three
workload profiles are rendered.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memory_gate.middleware import DEFAULT_PROTECT_PATTERNS  # noqa: E402
from memory_gate.review import ReviewItem, build_review_id, render_review_text  # noqa: E402

WINDOW_TOKENS = 200_000
FIRE_AT = 0.80
TARGET = int(WINDOW_TOKENS * FIRE_AT)
BUDGET_FRACTION = 0.02

# (tokens of AI reply, tokens of leftover non-dialogue) per user turn.
PROFILES = {
    "chat-heavy": (400, 0),
    "light-tools": (300, 3_000),
    "heavy-tools": (200, 15_000),
}

ARCHIVE = Path(__file__).resolve().parents[1] / ".scratch/gate-hard/archive.jsonl"
OUT = Path(__file__).resolve().parents[1] / ".scratch/scale_render"

# Compiled exactly as middleware.py does. An earlier version of this file left
# out IGNORECASE, which silently changed every English match and produced flag
# rates that did not correspond to the shipped behaviour.
PATTERNS = [re.compile(p, re.IGNORECASE) for p in DEFAULT_PROTECT_PATTERNS]


def is_flagged(text: str) -> bool:
    return any(p.search(text) for p in PATTERNS)


def load_real_turns():
    """The recorded run's user turns, plus how many of them the flags catch.

    Accuracy of the flag set is deliberately *not* computed here. Scoring needs
    a ground truth, and the keyword list this file used to carry counted any
    message mentioning 代号 as a constraint -- including two questions about
    codenames -- which put recall 7 points below the real figure. That job lives
    in scripts/score_patterns.py, with its corpora split so the answer means
    something.
    """
    rows = [json.loads(line) for line in
            ARCHIVE.read_text(encoding="utf-8").splitlines()]
    human = [r for r in rows if r["kind"] == "dialogue" and r["role"] == "human"]
    texts = [r["text"] for r in human]
    flag_rate = sum(1 for t in texts if is_flagged(t)) / len(texts)
    return texts, statistics.mean(r["tokens"] for r in human), flag_rate


def build_items(texts: list[str], tokens: int) -> list[ReviewItem]:
    """Same construction as MemoryGateMiddleware._run_gate, copied from source."""
    return [
        ReviewItem(
            ref=i + 1,
            id=f"synthetic-{i:05d}",
            role="human",
            pass_no=1,
            tokens=tokens,
            flagged=is_flagged(t),
            preview=t[:120],
        )
        for i, t in enumerate(texts)
    ]


def render(items, dropped_total, dialogue_tokens, budget, *, max_items=None):
    """The library's own renderer. Both columns come from here, so the
    comparison is of one code path at two settings, not of a hand-written
    mock-up against the real thing -- the mock-up version of this script
    drifted from the shipped format within a day of being written.

    ``max_items=None`` means the renderer's own default, read from the library
    rather than copied here, so the two cannot disagree.
    """
    request = {
        "review_id": build_review_id(items),
        "reason": "constraint_language",
        "dropped_total": dropped_total,
        "dropped_tokens": dialogue_tokens,
        "items": items,
        "omitted": dropped_total - len(items),
        "allowed": ["keep_selected", "keep_all", "confirm"],
        "budget_tokens": budget,
    }
    if max_items is None:
        return render_review_text(request)
    return render_review_text(request, max_items=max_items)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    texts, mean_tokens, flag_rate = load_real_turns()
    budget = int(TARGET * BUDGET_FRACTION)
    print(f"real user turns replayed : {len(texts)}")
    print(f"mean human turn tokens   : {mean_tokens:.1f}")
    print(f"DEFAULT_PROTECT_PATTERNS : flag rate {flag_rate:.0%}"
          "   (accuracy: scripts/score_patterns.py)")
    print(f"tokens leaving the window: {TARGET:,}   budget at {BUDGET_FRACTION:.0%}"
          f" = {budget:,}")
    print()
    print(f"{'profile':<13} {'user turns':>10} {'A lines':>8} {'A chars':>9} "
          f"{'B lines':>8} {'B chars':>9}")
    print("A = uncapped, what shipped before.  B = the library's own default cap.")
    for name, (ai_t, other_t) in PROFILES.items():
        per_turn = mean_tokens + ai_t + other_t
        n_turns = max(1, int(TARGET / per_turn))
        session = [texts[i % len(texts)] for i in range(n_turns)]
        items = build_items(session, round(mean_tokens))
        dropped_total = n_turns * 3
        dialogue_tokens = int(n_turns * mean_tokens)
        wall = render(items, dropped_total, dialogue_tokens, budget,
                      max_items=len(items))
        capped = render(items, dropped_total, dialogue_tokens, budget)
        (OUT / f"{name}_A_uncapped.txt").write_text(wall, encoding="utf-8")
        (OUT / f"{name}_B_capped.txt").write_text(capped, encoding="utf-8")
        print(f"{name:<13} {n_turns:>10,} {wall.count(chr(10)) + 1:>8,} {len(wall):>9,} "
              f"{capped.count(chr(10)) + 1:>8,} {len(capped):>9,}")
    print(f"\nfull renderings in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The review contract: payload types, text rendering, reply parsing.

Two deliberate decisions live here.

**The request payload is structured, never prose.** ``interrupt()`` carries a
``MemoryReviewRequest``. Rendering it to human-readable text is a separate,
optional step (``render_review_text``), so a host with its own approval UI can
ignore the renderer entirely. This mirrors ``HumanInTheLoopMiddleware``, whose
``HITLRequest`` is structured and whose ``description`` field is only a label.

**An unrecognised reply means "keep everything".** The parser is intentionally
dumb -- numbered refs only, no LLM, no fuzzy matching. When it cannot parse a
reply it returns ``mode="unrecognized"`` and the caller pins every listed item.
Guessing which messages a user meant to save is exactly the silent-failure
class this package exists to eliminate, so on ambiguity we fail closed.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal, TypedDict

from typing_extensions import NotRequired

__all__ = [
    "MemoryReviewRequest",
    "MemoryReviewResponse",
    "ReviewItem",
    "build_review_id",
    "parse_reply",
    "render_review_text",
]

DecisionMode = Literal["keep_selected", "keep_all", "confirm", "unrecognized"]


class ReviewItem(TypedDict):
    """One candidate for destruction, as shown to the reviewer."""

    ref: int
    """1-based number the user types back. Stable for a given review_id."""

    id: str
    """LangChain message id -- the handle used for pinning and later restore."""

    role: str
    pass_no: int
    tokens: int
    flagged: bool
    """True when this item matched a protect pattern, i.e. it looks like a
    durable constraint rather than chatter.

    The reviewer still decides. This only says *why* the item is on the list, so
    a human can confirm a shortlist instead of rereading the transcript -- and so
    a host can adopt a "pin the flagged ones" policy without writing its own
    classifier.
    """
    preview: str
    """Truncated verbatim text. Verbatim, not paraphrased: a paraphrase can
    omit an item, and an omitted item is one the user never got to save."""


def flagged_refs(request: "MemoryReviewRequest") -> list[int]:
    """Refs worth pinning under the "only keep constraints" policy."""
    return [i["ref"] for i in request["items"] if i.get("flagged")]


class MemoryReviewRequest(TypedDict):
    """Payload passed to ``interrupt()``."""

    review_id: str
    """Content fingerprint of ``items``. The middleware re-checks it after
    resume, because LangGraph replays the node from the top and a rebuilt list
    that differs from the one the user approved would be the worst possible
    bug: approving one thing and executing another."""

    reason: str
    dropped_total: int
    """Count of messages leaving the window, dialogue and tool alike."""

    dropped_tokens: int
    """Tokens of the dialogue items listed below."""

    items: list[ReviewItem]
    omitted: int
    """Messages being dropped that are *not* listed (tool results, etc.).
    Surfaced explicitly so the reviewer knows the list is not the whole story."""

    allowed: list[str]
    omitted_previews: NotRequired[list[str]]
    budget_tokens: NotRequired[int]
    """``protect_budget_tokens`` of the gate that built this request.

    Carried so the rendering can say what ``keep all`` will actually cost and
    which end of the conversation the overflow falls off. Without it the
    reviewer approves a pin set blind and discovers the oldest rules went
    missing several turns later.
    """


class MemoryReviewResponse(TypedDict):
    """What ``Command(resume=...)`` must carry."""

    mode: DecisionMode
    keep_refs: list[int]
    raw: NotRequired[str]
    """The reviewer's literal reply, kept for the audit trail."""


def build_review_id(items: list[ReviewItem]) -> str:
    """Fingerprint the candidate set so a replay can be detected."""
    blob = json.dumps(
        [[i["id"], i["tokens"], i["preview"]] for i in items],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


_KEEP_RE = re.compile(
    r"^\s*(?:keep|retain|save|保留|留住|保住)\s*[:：]?\s*(.+?)\s*$",
    re.IGNORECASE,
)
_ALL_RE = re.compile(
    r"^\s*(?:keep\s*all|save\s*all|all|全部保留|全保留|都保留|都留下)\s*$",
    re.IGNORECASE,
)
_CONFIRM_RE = re.compile(
    r"^\s*(?:confirm|confirmed|ok|okay|yes|y|go|proceed|压缩|确认|继续|好)\s*$",
    re.IGNORECASE,
)
_NUMS_RE = re.compile(r"\d+")


def parse_reply(reply: Any, valid_refs: set[int]) -> MemoryReviewResponse:
    """Parse a reviewer reply into a decision. Never raises.

    Accepts a ``MemoryReviewResponse`` dict (structured hosts) or a bare
    string / list of ints (plain chat hosts). Anything ambiguous becomes
    ``unrecognized``, which the caller must treat as "keep everything".
    """
    if isinstance(reply, dict):
        mode = reply.get("mode")
        raw_refs = reply.get("keep_refs", [])
        refs = [int(r) for r in raw_refs if isinstance(r, (int, str)) and str(r).isdigit()]
        if mode == "keep_all":
            return {"mode": "keep_all", "keep_refs": sorted(valid_refs), "raw": str(reply)}
        if mode == "confirm":
            return {"mode": "confirm", "keep_refs": [], "raw": str(reply)}
        if mode == "keep_selected" and refs and set(refs) <= valid_refs:
            return {"mode": "keep_selected", "keep_refs": sorted(refs), "raw": str(reply)}
        return {"mode": "unrecognized", "keep_refs": [], "raw": str(reply)}

    if isinstance(reply, (list, tuple, set)):
        requested = {
            int(r) for r in reply if isinstance(r, int) or (isinstance(r, str) and r.isdigit())
        }
        if requested and requested <= valid_refs:
            return {"mode": "keep_selected", "keep_refs": sorted(requested), "raw": repr(reply)}
        return {"mode": "unrecognized", "keep_refs": [], "raw": repr(reply)}

    if isinstance(reply, int):
        if reply in valid_refs:
            return {"mode": "keep_selected", "keep_refs": [reply], "raw": str(reply)}
        return {"mode": "unrecognized", "keep_refs": [], "raw": str(reply)}

    text = str(reply).strip()
    if not text:
        return {"mode": "unrecognized", "keep_refs": [], "raw": text}

    if _ALL_RE.match(text):
        return {"mode": "keep_all", "keep_refs": sorted(valid_refs), "raw": text}

    m = _KEEP_RE.match(text)
    target = m.group(1) if m else text
    requested = {int(n) for n in _NUMS_RE.findall(target)}
    if requested and requested <= valid_refs:
        return {"mode": "keep_selected", "keep_refs": sorted(requested), "raw": text}
    if requested:
        # The reviewer pointed at something not on the list. Acting on the part
        # that *was* valid would be a silent partial read of an ambiguous
        # instruction, so treat it the same as "keep 99": unrecognised.
        return {"mode": "unrecognized", "keep_refs": [], "raw": text}

    if _CONFIRM_RE.match(text):
        return {"mode": "confirm", "keep_refs": [], "raw": text}

    # "keep 百合花" -- the user pointed at content, not a number. We cannot
    # resolve that safely, so we refuse to guess and keep everything.
    return {"mode": "unrecognized", "keep_refs": [], "raw": text}


def render_review_text(req: MemoryReviewRequest, *, preview_width: int = 72,
                       max_items: int = 40) -> str:
    """Render a request as a plain-chat review block. No LLM involved.

    ``max_items`` caps what is *printed*, never what is pinnable: the payload
    keeps every candidate so refs stay addressable and ``keep all`` still covers
    the rows that did not fit on screen. Capping by the protect budget instead
    would hide exactly the oldest messages, which is where a rule stated once at
    the start of a long session lives.

    Rows are ordered rules-first, then oldest-first. A production-scale review
    was measured at 403 lines / 27K characters for a 390-turn chat session
    (``scripts/render_at_scale.py``), which nobody reads; ordering plus a cap is
    what makes the block skimmable, and the default action is deliberately one
    word so that not reading it is still safe.
    """
    items = req["items"]
    shown = sorted(items, key=lambda i: (not i.get("flagged"), i["ref"]))[:max_items]
    hidden = len(items) - len(shown)
    flagged_n = sum(1 for i in items if i.get("flagged"))
    mixed_roles = len({i["role"] for i in shown}) > 1

    lines = [
        f"Compaction will drop {req['dropped_total']} messages."
        f" {len(items)} of them are dialogue you can keep"
        f" ({req['dropped_tokens']} tokens).",
        "The rest survive only as a lossy summary."
        " Nothing is deleted -- all of it stays in the archive.",
    ]

    budget = req.get("budget_tokens")
    if budget and req["dropped_tokens"] > budget:
        lines.append(
            f"keep all would pin {req['dropped_tokens']} tokens but the budget is"
            f" {budget}: the oldest {req['dropped_tokens'] - budget} tokens stay in the"
            " archive and are not re-sent to the model."
        )
    if hidden > 0:
        lines.append(
            f"Showing {len(shown)} of {len(items)} -- rules first, then oldest."
            f" The {hidden} not shown are still pinnable and are covered by keep all."
        )
    lines.append("")

    for item in shown:
        preview = item["preview"].replace("\n", " ")
        if len(preview) > preview_width:
            preview = preview[: preview_width - 1] + "\u2026"
        mark = "*" if item.get("flagged") else " "
        role = f" {item['role']:<9}" if mixed_roles else ""
        lines.append(f" [{item['ref']:>4}]{mark} {item['tokens']:>4} tok {role} {preview}")

    if flagged_n:
        lines += [
            "",
            f" * matched a protect pattern: {flagged_n} of {len(items)} look like a"
            " standing rule.",
            "   The patterns miss some real rules, so an unmarked line is not a line"
            " you can afford to lose.",
        ]
    if req.get("omitted"):
        lines += [
            "",
            f" {req['omitted']} further messages are being dropped and are not listed"
            " (tool results, and assistant replies unless pin_roles is widened).",
        ]
    # Built from refs actually on screen: a hard-coded "keep 3,8,11" invited
    # replies naming items that do not exist, which parse_reply cannot resolve
    # and therefore treats as keep-everything.
    example = ",".join(str(i["ref"]) for i in sorted(shown[:3], key=lambda i: i["ref"])) or "1"
    lines += [
        "",
        f"  keep all     pin all {len(items)}, budget permitting   <- the usual answer",
        f"  keep {example}  pin only these refs",
        "  confirm      compact and pin nothing new",
    ]
    return "\n".join(lines)

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
    turn: int
    tokens: int
    preview: str
    """Truncated verbatim text. Verbatim, not paraphrased: a paraphrase can
    omit an item, and an omitted item is one the user never got to save."""


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


def render_review_text(req: MemoryReviewRequest, *, preview_width: int = 72) -> str:
    """Render a request as a plain-chat review block. No LLM involved."""
    lines = [
        f"About to compact {req['dropped_total']} messages "
        f"({req['dropped_tokens']} tokens of dialogue).",
        "After compaction these survive only as a lossy summary.",
        "Reply with refs to pin them permanently:",
        "",
    ]
    for item in req["items"]:
        preview = item["preview"].replace("\n", " ")
        if len(preview) > preview_width:
            preview = preview[: preview_width - 1] + "\u2026"
        lines.append(
            f" [{item['ref']:>2}] #{item['id']:<10} {item['tokens']:>5} tok  "
            f"{item['role']:<9} {preview}"
        )
    if req.get("omitted"):
        lines.append("")
        lines.append(
            f" {req['omitted']} more messages (tool results, non-dialogue) are also"
            " being dropped and are not listed here."
        )
    lines += [
        "",
        "  keep 1,3,5   pin the listed refs",
        "  keep all     pin everything listed",
        "  confirm      compact without pinning",
    ]
    return "\n".join(lines)

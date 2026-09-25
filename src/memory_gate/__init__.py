"""memory-gate: a fail-closed human gate in front of LLM context compaction."""

from __future__ import annotations

from .archive import Archive, ArchiveWriteError, GateError, Record
from .middleware import DEFAULT_PROTECT_PATTERNS, MemoryGateMiddleware
from .review import (
    MemoryReviewRequest,
    MemoryReviewResponse,
    ReviewItem,
    flagged_refs,
    parse_reply,
    render_review_text,
)

__version__ = "0.2.0"

__all__ = [
    "Archive",
    "ArchiveWriteError",
    "DEFAULT_PROTECT_PATTERNS",
    "GateError",
    "MemoryGateMiddleware",
    "MemoryReviewRequest",
    "MemoryReviewResponse",
    "Record",
    "ReviewItem",
    "__version__",
    "flagged_refs",
    "parse_reply",
    "render_review_text",
]

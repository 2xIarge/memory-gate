"""On-disk archive and protected-item store.

Design rules this module exists to enforce:

1. **Archiving is unconditional and happens before anything can be deleted.**
   The archive is a superset of what any compaction middleware might drop.
   Over-archiving costs disk; under-archiving loses data permanently.

2. **Every write failure raises.** ``memory_gate`` never degrades silently.
   If we cannot guarantee the original text is on disk, we must not let
   compaction proceed -- the caller's ``before_model`` raises, the graph run
   fails, and the messages stay in state. That is the same fail-loud choice
   LangChain made for langchain#38867.

3. **Append-only JSON Lines.** One message per line, so a partial write can
   never corrupt records that are already on disk, and ``grep`` works.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

__all__ = [
    "Archive",
    "ArchiveWriteError",
    "GateError",
    "Record",
]


class GateError(RuntimeError):
    """Base class for every error raised by memory-gate."""


class ArchiveWriteError(GateError):
    """The archive could not be written.

    Raised instead of continuing. Compaction must not proceed when the
    original text is not safely on disk.
    """


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class Record:
    """One archived message.

    ``id`` is the LangChain message id. ``SummarizationMiddleware`` calls
    ``_ensure_message_ids`` as the first thing it does in ``before_model``, so
    by the time compaction is possible every message already has a stable id.
    We rely on that as the handle for "keep this one" and for later restore.
    """

    id: str
    role: str
    turn: int
    ts: str
    tokens: int
    text: str
    kind: str = "dialogue"
    """``dialogue`` | ``tool`` | ``summary`` | ``other``.

    Only ``dialogue`` is ever put in front of the reviewer. Everything is
    archived regardless -- the archive is a superset on purpose.
    """

    protected: bool = False
    """True once the user has pinned this record."""

    def to_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_line(cls, line: str) -> Record:
        data = json.loads(line)
        fields = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in fields})

    @property
    def is_dialogue(self) -> bool:
        return self.kind == "dialogue"


class Archive:
    """Append-only JSONL store plus the user's protected list."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.archive_path = self.root / "archive.jsonl"
        self.keep_path = self.root / "keep.jsonl"

    # ---------------------------------------------------------------- io

    def _ensure_root(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # pragma: no cover - environment dependent
            raise ArchiveWriteError(
                f"memory-gate cannot create archive dir {self.root!s}: {exc}"
            ) from exc

    @staticmethod
    def _append_lines(path: Path, lines: list[str]) -> None:
        """Append and fsync. Durability is the entire point of this package."""
        if not lines:
            return
        try:
            with path.open("a", encoding="utf-8") as fh:
                for line in lines:
                    fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        except OSError as exc:
            raise ArchiveWriteError(
                f"memory-gate could not write {path!s}: {exc}"
            ) from exc

    @staticmethod
    def _read(path: Path) -> list[Record]:
        if not path.exists():
            return []
        out: list[Record] = []
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(Record.from_line(line))
                except (json.JSONDecodeError, TypeError) as exc:
                    # A corrupt line must never be skipped silently: skipping it
                    # means believing a message is archived when it is not.
                    raise GateError(
                        f"memory-gate: unreadable record at {path!s}:{lineno}: {exc}"
                    ) from exc
        return out

    # ----------------------------------------------------------- queries

    def seen_ids(self) -> set[str]:
        """Ids already on disk, in either file. Used to make archiving idempotent."""
        return {r.id for r in self._read(self.archive_path)} | {
            r.id for r in self._read(self.keep_path)
        }

    def protected(self) -> list[Record]:
        """Everything the user has pinned, in pin order."""
        return self._read(self.keep_path)

    def protected_ids(self) -> set[str]:
        return {r.id for r in self.protected()}

    def get(self, message_id: str) -> Record | None:
        for rec in (*self.protected(), *self._read(self.archive_path)):
            if rec.id == message_id:
                return rec
        return None

    def search(self, needle: str) -> list[Record]:
        """Substring search over archived text, protected items first."""
        low = needle.lower()
        hits = [r for r in self.protected() if low in r.text.lower()]
        seen = {r.id for r in hits}
        hits += [
            r
            for r in self._read(self.archive_path)
            if low in r.text.lower() and r.id not in seen
        ]
        return hits

    # ------------------------------------------------------------ writes

    def append(self, records: Iterable[Record]) -> int:
        """Archive records, skipping ids already on disk. Returns count written."""
        self._ensure_root()
        known = self.seen_ids()
        fresh = [r for r in records if r.id not in known]
        self._append_lines(self.archive_path, [r.to_line() for r in fresh])
        return len(fresh)

    def protect(self, records: Iterable[Record]) -> int:
        """Pin records. Idempotent by id; re-pinning does not duplicate."""
        self._ensure_root()
        known = self.protected_ids()
        fresh = [
            Record(**{**asdict(r), "protected": True})
            for r in records
            if r.id not in known
        ]
        self._append_lines(self.keep_path, [r.to_line() for r in fresh])
        return len(fresh)

    # -------------------------------------------------------------- misc

    def stats(self) -> dict[str, int]:
        arch = self._read(self.archive_path)
        keep = self._read(self.keep_path)
        return {
            "archived": len(arch),
            "archived_tokens": sum(r.tokens for r in arch),
            "protected": len(keep),
            "protected_tokens": sum(r.tokens for r in keep),
        }

    def mark_run(self, payload: dict[str, object]) -> None:
        """Append an observability line.

        A gate that ran must leave a trace, and a gate that did *not* run must
        be distinguishable from one that ran and found nothing. This is the
        lesson of langchain#39247, where an approval gate was silently disabled
        by a bad config and nothing recorded the fact.
        """
        self._ensure_root()
        line = json.dumps(
            {"type": "gate_run", "ts": _utcnow(), **payload},
            ensure_ascii=False,
            sort_keys=True,
        )
        self._append_lines(self.root / "runs.jsonl", [line])


def utcnow() -> str:
    """Public wrapper so middleware and tests share one clock format."""
    return _utcnow()

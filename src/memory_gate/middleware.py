"""``MemoryGateMiddleware`` -- a fail-closed human gate in front of compaction.

Placement
---------
Put this **first** in the middleware list::

    middleware=[MemoryGateMiddleware(...), SummarizationMiddleware(...)]

Two facts about ``langchain.agents.factory`` make that ordering meaningful
(both verified against langchain 1.4.1):

* ``before_model`` hooks become their own graph nodes named
  ``<middleware>.before_model``, chained in list order, with
  ``middleware_w_before_model[0]`` as the entry node. So being first means we
  see every message *before* ``SummarizationMiddleware`` can delete it.
* ``wrap_model_call`` handlers compose "first = outermost". So being first also
  means our injection is the last thing applied before the model is called,
  i.e. after compaction has already rewritten the history.

Because ``before_model`` is a separate node and not inside the
``wrap_model_call`` chain, an ``interrupt()`` raised here cannot be swallowed
by ``ModelRetryMiddleware`` / ``ModelFallbackMiddleware`` -- the defect class
described in langchain#38837. Staying first in the list keeps it that way.

Failure behaviour
-----------------
Every failure path raises rather than continuing. If the archive cannot be
written, compaction must not happen: an exception in this node aborts the run
and the messages stay in state. That is the same fail-loud trade-off LangChain
adopted for langchain#38867, and the opposite of the silent gate disablement
fixed in langchain#39247.
"""

from __future__ import annotations

import re
import uuid
import warnings
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from typing_extensions import Annotated, NotRequired, TypedDict, override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    PrivateStateAttr,
    ResponseT,
    StateT,
)

from .archive import Archive, GateError, Record, utcnow
from .review import (
    MemoryReviewRequest,
    MemoryReviewResponse,
    ReviewItem,
    build_review_id,
    parse_reply,
    render_review_text,
)

__all__ = ["MemoryGateMiddleware", "DEFAULT_PROTECT_PATTERNS"]

try:  # prefer langchain's own estimator so our numbers match the gate we guard
    from langchain_core.messages.utils import count_tokens_approximately

    def _default_token_counter(messages: Sequence[Any]) -> int:
        return int(count_tokens_approximately(messages))

except Exception:  # pragma: no cover - very old langchain-core

    def _default_token_counter(messages: Sequence[Any]) -> int:
        return sum(len(_text_of(m)) for m in messages) // 4


DEFAULT_PROTECT_PATTERNS: tuple[str, ...] = (
    # English constraint markers
    r"\bmust\b", r"\bnever\b", r"\balways\b", r"\bdo not\b", r"\bdon't\b",
    r"\bforbidden\b", r"\brequire[sd]?\b", r"\bconstraint\b", r"\bpolicy\b",
    r"\bcomply\b", r"\bcompliance\b", r"\bonly use\b", r"\bremember\b",
    # Chinese constraint markers
    r"必须", r"不允许", r"不得", r"禁止", r"一律", r"务必", r"口径",
    r"约定", r"记住", r"千万", r"只能", r"不要", r"别",
)
"""Heuristics that flag an utterance as a likely durable constraint.

These only decide *whether to ask*. They never decide what is kept, and a miss
is not fatal: the message is still archived unconditionally, so it can be
recovered afterwards with ``Archive.search``.
"""


def _text_of(message: BaseMessage) -> str:
    """Flatten ``message.content`` (str or block list) to plain text."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
                elif block.get("type") == "tool_use":
                    parts.append(f"[tool_use {block.get('name', '')}]")
        return "\n".join(p for p in parts if p)
    return str(content)


def _kind_of(message: BaseMessage) -> str:
    """Classify a message. Only ``dialogue`` is ever shown to the reviewer.

    Tool traffic is deliberately excluded: it is reproducible from disk or by
    re-running the call, and it is the bulk of the tokens. Leaving it to the
    host's own compaction keeps the review list short enough to actually read.

    Compaction summaries are excluded too. ``SummarizationMiddleware`` injects
    its summary as a ``HumanMessage`` tagged with
    ``additional_kwargs={"lc_source": "summarization"}`` -- so without this
    check the *summary itself* shows up in the next round's review list and the
    user is asked whether to preserve a machine-generated paraphrase. That is
    not the decision this gate exists to offer. Both kinds are still archived;
    only the review list is filtered.
    """
    if getattr(message, "additional_kwargs", None):
        if message.additional_kwargs.get("lc_source") == "summarization":
            return "summary"
    if isinstance(message, ToolMessage):
        return "tool"
    if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
        return "tool"
    if isinstance(message, SystemMessage):
        return "other"
    if isinstance(message, (HumanMessage, AIMessage)):
        return "dialogue" if _text_of(message).strip() else "other"
    return "other"


class _MemoryGateState(AgentState[ResponseT]):
    """State extension carrying the last gate outcome.

    Private, following ``HumanInTheLoopMiddleware``'s use of ``PrivateStateAttr``
    so it stays out of the agent's input/output schemas.
    """

    memory_gate: NotRequired[Annotated[dict[str, Any], PrivateStateAttr]]


class MemoryGateMiddleware(AgentMiddleware[StateT, ContextT, ResponseT]):
    """Archive everything, then ask before letting dialogue be compacted away."""

    state_schema = _MemoryGateState  # type: ignore[assignment]

    def __init__(
        self,
        *,
        archive_dir: str = ".memory-gate",
        summarization: Any | None = None,
        trigger: Any | None = None,
        keep_messages: int | None = None,
        review_threshold_tokens: int = 400,
        protect_patterns: Sequence[str] | None = None,
        pin_roles: Sequence[str] = ("human",),
        preview_chars: int = 120,
        protect_budget_tokens: int = 1200,
        inject: bool = True,
        enabled: bool = True,
        token_counter: Callable[[Sequence[Any]], int] | None = None,
    ) -> None:
        """
        Args:
            archive_dir: Where ``archive.jsonl`` / ``keep.jsonl`` / ``runs.jsonl`` live.
            summarization: Optional ``SummarizationMiddleware`` instance. Only its
                *public* ``trigger``, ``keep`` and ``token_counter`` attributes are
                read, so we stay decoupled from its private cutoff internals.
            trigger: Trigger to use instead of reading ``summarization.trigger``.
                Accepts the same shapes the library accepts.
            keep_messages: Messages at the tail that compaction will preserve.
                Defaults to ``summarization.keep`` when it is message-based, else 20
                (the langchain 1.4.1 default).
            review_threshold_tokens: Pause and ask once this many dialogue tokens
                are about to leave the window, even if no pattern matched.
            protect_patterns: Regexes marking a likely durable constraint.
            pin_roles: Roles whose messages may be offered for pinning. Defaults
                to human only: constraints are stated by the user, and the
                assistant's restatement of the same rule is redundant text that
                competes for the same protect budget it is meant to protect.
                Widen it if you want replies pinned too.
            preview_chars: Verbatim preview length per review item.
            protect_budget_tokens: Hard ceiling on pinned text injected into every
                model call. Without one the injection raises the reported token count,
                which trips the compaction trigger, which causes another review, which
                pins more -- a runaway. Measured: 2 compaction cycles became 31.
                Overflow is omitted and declared, never silently dropped.
            inject: Re-inject pinned items on every model call.
            enabled: Kill switch; a disabled gate archives and injects nothing.
            token_counter: Override the estimator.

        Raises:
            ValueError: When neither ``summarization`` nor ``trigger`` is given.
                A gate that can never fire is the silent-disablement bug that
                langchain#39247 exists to prevent, so we refuse the config
                instead of shipping a no-op.
        """
        super().__init__()
        if summarization is None and trigger is None:
            raise ValueError(
                "memory-gate: pass summarization=<the SummarizationMiddleware you "
                "are guarding> or an explicit trigger=(...). Without one the gate "
                "can never tell whether compaction is imminent, and a review gate "
                "that never fires is worse than no gate -- it is silently useless "
                "(see langchain#39247 for exactly this failure mode)."
            )
        if review_threshold_tokens < 0:
            raise ValueError(
                "memory-gate: review_threshold_tokens must be >= 0; a negative "
                "value would silently disable the review gate."
            )
        if preview_chars <= 0:
            raise ValueError("memory-gate: preview_chars must be > 0.")
        self.pin_roles = tuple(pin_roles)
        if not self.pin_roles:
            raise ValueError(
                "memory-gate: pin_roles is empty, so nothing could ever be offered "
                "for pinning -- a review gate with no candidates never fires."
            )
        if protect_budget_tokens < 100:
            raise ValueError(
                "memory-gate: protect_budget_tokens must be >= 100; a tiny budget "
                "injects nothing while still looking armed -- the silent no-op "
                "langchain#39247 exists to prevent."
            )

        self.archive = Archive(archive_dir)
        self.review_threshold_tokens = review_threshold_tokens
        self.preview_chars = preview_chars
        self.protect_budget_tokens = protect_budget_tokens
        self.inject = inject
        self.enabled = enabled
        self._summarization = summarization
        self._trigger = self._normalize_trigger(
            trigger if trigger is not None else getattr(summarization, "trigger", None)
        )
        self._fraction_limit = self._resolve_fraction_limit(summarization)
        self._keep_messages = self._resolve_keep_messages(summarization, keep_messages)
        if summarization is None and keep_messages is None:
            # Observed in the wild: a gate configured with only a trigger silently
            # assumed keep=20 while the middleware it guarded kept 4. The computed
            # window was then always empty, so it never reviewed anything and never
            # said so.
            warnings.warn(
                "memory-gate: no `summarization` object was passed, so keep_messages "
                f"defaults to {self._keep_messages}. If the middleware you guard keeps "
                "a different number, this gate mis-times its reviews and may never "
                "fire at all -- pass summarization=<instance> or keep_messages=<n>.",
                stacklevel=2,
            )
        self._patterns = [
            re.compile(p, re.IGNORECASE)
            for p in (protect_patterns if protect_patterns is not None else DEFAULT_PROTECT_PATTERNS)
        ]
        self._token_counter = (
            token_counter
            or (getattr(summarization, "token_counter", None) if summarization is not None else None)
            or _default_token_counter
        )

    # ------------------------------------------------------------ config

    @staticmethod
    def _normalize_trigger(trigger: Any) -> list[dict[str, float]]:
        """Reduce the library's trigger forms to a list of AND-clauses (OR across).

        Unknown shapes degrade to "immediately met", which asks too often rather
        than silently never asking.
        """
        if trigger is None:
            return []
        if isinstance(trigger, tuple) and len(trigger) == 2:
            return [{str(trigger[0]): float(trigger[1])}]
        if isinstance(trigger, dict):
            return [{str(k): float(v) for k, v in trigger.items()}]
        if isinstance(trigger, (list, tuple)):
            out: list[dict[str, float]] = []
            for item in trigger:
                out.extend(MemoryGateMiddleware._normalize_trigger(item))
            return out
        return [{"messages": 0.0}]

    @staticmethod
    def _resolve_fraction_limit(summarization: Any) -> int | None:
        """Best-effort max input window, only for a ``fraction`` trigger."""
        model = getattr(summarization, "model", None)
        profile = getattr(model, "profile", None)
        value = None
        if isinstance(profile, dict):
            value = profile.get("max_input_tokens")
        else:
            value = getattr(profile, "max_input_tokens", None)
        try:
            return int(value) if value else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _resolve_keep_messages(summarization: Any, explicit: int | None) -> int:
        if explicit is not None:
            if explicit < 0:
                raise ValueError("memory-gate: keep_messages must be >= 0.")
            return explicit
        keep = getattr(summarization, "keep", None)
        if isinstance(keep, tuple) and len(keep) == 2 and keep[0] == "messages":
            return int(keep[1])
        # Token- or fraction-based keep windows have no message-count equivalent,
        # so fall back to the library default rather than guessing.
        return 20

    def _is_risky(self, text: str) -> bool:
        return any(p.search(text) for p in self._patterns)

    @staticmethod
    def _reported_tokens(messages: Sequence[AnyMessage]) -> int:
        """``usage_metadata.total_tokens`` of the newest AIMessage, else -1.

        ``SummarizationMiddleware._should_summarize`` fires a ``tokens`` clause
        when the approximate count crosses the threshold **or** the model's own
        reported total does. Ignoring that second half -- as this file originally
        did -- makes the gate believe compaction happens later than it does, and
        content then leaves the window without ever being reviewed. Observed with
        a real model: a constraint stated on turn 0 was compacted away unasked.
        """
        for message in reversed(list(messages)):
            if isinstance(message, AIMessage):
                usage = getattr(message, "usage_metadata", None)
                if not usage:
                    return -1
                try:
                    return int(usage.get("total_tokens", -1))
                except (TypeError, ValueError, AttributeError):
                    return -1
        return -1

    def _clause_met(self, clause: dict[str, float], n_messages: int, total_tokens: int,
                    reported_tokens: int) -> bool:
        if not clause:
            return False
        for key, value in clause.items():
            if key == "messages":
                if n_messages < value:
                    return False
            elif key == "tokens":
                if total_tokens < value and reported_tokens < value:
                    return False
            elif key == "fraction":
                if self._fraction_limit is None:
                    continue  # cannot evaluate conservatively; treat as met
                threshold = value * self._fraction_limit
                if total_tokens < threshold and reported_tokens < threshold:
                    return False
            # unknown key: ignored, i.e. treated as met (asks rather than skips)
        return True

    def _compaction_is_imminent(self, messages: Sequence[AnyMessage],
                                total_tokens: int) -> bool:
        """Would the guarded middleware actually compact on this turn?

        Without this the gate asks about deletions that are not happening yet,
        so the reviewer is interrupted nearly every turn -- the fastest route to
        "make it stop" and to the gate being uninstalled.

        The library additionally requires a model-provider match before it trusts
        reported tokens. We deliberately do not check it: a false alarm costs one
        extra prompt (and identical candidate sets are never re-asked anyway),
        while a missed review is silent, unrecoverable loss. Raise
        ``review_threshold_tokens`` if a host sees too many alarms.
        """
        n_messages = len(messages)
        reported = self._reported_tokens(messages)
        return any(self._clause_met(c, n_messages, total_tokens, reported)
                   for c in self._trigger)

    def _ensure_ids(self, messages: list[AnyMessage]) -> None:
        """Mirror ``SummarizationMiddleware._ensure_message_ids``.

        A message without an id cannot be pinned or restored, so it gets one
        here rather than being archived anonymously.
        """
        for message in messages:
            if getattr(message, "id", None) is None:
                message.id = str(uuid.uuid4())

    def _to_record(self, message: AnyMessage, pass_no: int) -> Record:
        text = _text_of(message)
        role = getattr(message, "type", None) or message.__class__.__name__.lower()
        return Record(
            id=str(message.id),
            role=str(role),
            pass_no=pass_no,
            ts=utcnow(),
            tokens=int(self._token_counter([message])),
            text=text,
            kind=_kind_of(message),
        )

    # ------------------------------------------------------ the gate itself

    def _run_gate(self, state: Any) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        messages: list[AnyMessage] = list(state.get("messages") or [])
        if not messages:
            return None

        self._ensure_ids(messages)

        # ``turn`` used to be the message's index in the current list. Once the
        # first compaction lands, that list stays pinned near the keep window, so
        # the number stopped being a turn at all -- it read 4, 5, 6 forever. A
        # counter carried in private state is monotone and names what it counts.
        carried = state.get("memory_gate") if isinstance(state, dict) else None
        pass_no = (int(carried.get("pass_no", 0)) + 1) if isinstance(carried, dict) else 1

        def outcome(extra: dict[str, Any]) -> dict[str, Any]:
            return {"memory_gate": {"pass_no": pass_no, **extra}}

        keep_n = self._keep_messages
        if len(messages) <= keep_n:
            self.archive.mark_run(
                {"reviewed": False, "reason": "within_keep_window", "messages": len(messages)}
            )
            return outcome({"reviewed": False, "reason": "within_keep_window"})

        zone = messages[: len(messages) - keep_n]
        records = [self._to_record(m, pass_no) for m in zone]

        # Unconditional, idempotent, and before anything can be deleted.
        written = self.archive.append(records)

        total_tokens = int(self._token_counter(messages))
        if not self._compaction_is_imminent(messages, total_tokens):
            self.archive.mark_run(
                {"reviewed": False, "reason": "compaction_not_imminent",
                 "archived": written, "zone": len(zone), "messages": len(messages),
                 "total_tokens": total_tokens,
                 "reported_tokens": self._reported_tokens(messages)}
            )
            return outcome({"reviewed": False, "reason": "compaction_not_imminent"})

        protected_ids = self.archive.protected_ids()
        candidates = [
            r for r in records
            if r.is_dialogue and r.role in self.pin_roles and r.id not in protected_ids
        ]
        if not candidates:
            self.archive.mark_run(
                {"reviewed": False, "reason": "no_unprotected_dialogue",
                 "archived": written, "zone": len(zone)}
            )
            return outcome({"reviewed": False, "reason": "no_unprotected_dialogue"})

        risky = [r for r in candidates if self._is_risky(r.text)]
        dialogue_tokens = sum(r.tokens for r in candidates)
        should_review = bool(risky) or dialogue_tokens >= self.review_threshold_tokens
        if not should_review:
            self.archive.mark_run(
                {"reviewed": False, "reason": "below_threshold", "archived": written,
                 "zone": len(zone), "dialogue_tokens": dialogue_tokens}
            )
            return outcome({"reviewed": False, "reason": "below_threshold"})

        items: list[ReviewItem] = [
            ReviewItem(
                ref=i + 1,
                id=r.id,
                role=r.role,
                pass_no=r.pass_no,
                tokens=r.tokens,
                flagged=self._is_risky(r.text),
                preview=r.text[: self.preview_chars],
            )
            for i, r in enumerate(candidates)
        ]
        review_id = build_review_id(items)

        if (
            isinstance(carried, dict)
            and carried.get("reviewed")
            and carried.get("review_id") == review_id
        ):
            # The reviewer already answered for exactly this candidate set. Asking
            # again would be pure nagging; the pinning decision is persisted on
            # disk, so skipping here loses nothing.
            self.archive.mark_run(
                {"reviewed": False, "reason": "already_reviewed", "review_id": review_id,
                 "archived": written, "zone": len(zone)}
            )
            return outcome({"reviewed": False, "reason": "already_reviewed",
                            "review_id": review_id})

        request: MemoryReviewRequest = {
            "review_id": review_id,
            "reason": "constraint_language" if risky else "token_threshold",
            "dropped_total": len(zone),
            "dropped_tokens": dialogue_tokens,
            "items": items,
            "omitted": len(zone) - len(candidates),
            "allowed": ["keep_selected", "keep_all", "confirm"],
        }

        payload = {"type": "memory_gate_review", **request, "text": render_review_text(request)}
        reply = interrupt(payload)

        # Replay guard: the node re-executes from the top on resume, so the list
        # is rebuilt. If it no longer matches what the reviewer approved, stop --
        # approving one set and compacting another is the worst outcome here.
        echoed = reply.get("review_id") if isinstance(reply, dict) else None
        if echoed is not None and echoed != request["review_id"]:
            raise GateError(
                f"memory-gate: review set changed after interrupt "
                f"(approved {echoed}, rebuilt {request['review_id']}). Refusing to compact."
            )

        decision: MemoryReviewResponse = parse_reply(reply, {i["ref"] for i in items})
        mode = decision["mode"]
        if mode == "confirm":
            keep_refs: list[int] = []
        elif mode in ("keep_all", "unrecognized"):
            # "unrecognized" pins everything. We will not guess which messages a
            # human meant to save.
            keep_refs = [i["ref"] for i in items]
        else:
            keep_refs = decision["keep_refs"]

        chosen = [r for r, i in zip(candidates, items) if i["ref"] in set(keep_refs)]
        pinned = self.archive.protect(chosen)

        run = {
            "pass_no": pass_no,
            "reviewed": True,
            "review_id": request["review_id"],
            "reason": request["reason"],
            "mode": mode,
            "archived": written,
            "zone": len(zone),
            "dialogue_tokens": dialogue_tokens,
            "pinned": pinned,
            "total_protected": len(self.archive.protected_ids()),
        }
        self.archive.mark_run(run)
        return {"memory_gate": run}

    # ------------------------------------------------------------- injection

    def _injection_block(self, records: Sequence[Record]) -> str:
        lines = [
            "<memory_gate>",
            "The following are VERBATIM user/assistant statements that the user",
            "explicitly pinned before context compaction. They outrank any summary",
            "of earlier conversation. Treat them as still in force.",
        ]
        for r in records:
            body = r.text.strip().replace("\n", " ")
            lines.append(f"- [{r.role} #{r.id}] {body}")
        lines.append("</memory_gate>")
        return "\n".join(lines)

    def protected_view(self) -> tuple[list[Record], list[Record]]:
        """Split the pinned set into (injected, omitted) under the budget.

        Newest first: a constraint stated 300 turns ago is likelier stale than one
        stated 3 turns ago. Nothing is deleted -- overflow stays pinned on disk and
        the omission is declared inside the block, so the model can say "I no longer
        have X in context" instead of quietly guessing at it.
        """
        records = self.archive.protected()
        kept: list[Record] = []
        used = 0
        for record in reversed(records):
            if used + record.tokens > self.protect_budget_tokens:
                break
            kept.append(record)
            used += record.tokens
        kept.reverse()
        included = {r.id for r in kept}
        return kept, [r for r in records if r.id not in included]

    def _apply_injection(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        if not (self.enabled and self.inject):
            return request
        kept, omitted = self.protected_view()
        if not kept and not omitted:
            return request
        present = {str(getattr(m, "id", "")) for m in request.messages}
        kept = [r for r in kept if r.id not in present]
        omitted = [r for r in omitted if r.id not in present]
        if not kept and not omitted:
            return request  # still verbatim in the window; no need to duplicate

        block = self._injection_block(kept)
        if omitted:
            block = block.replace(
                "</memory_gate>",
                f" {len(omitted)} further pinned statements are NOT in this context "
                "(the protect budget is full); they remain on disk and can be "
                "retrieved. Say so rather than guessing at them."
                + "\n</memory_gate>",
            )
        existing = getattr(request, "system_message", None)
        base = _text_of(existing) if existing is not None else ""
        merged = f"{base}\n\n{block}".strip() if base.strip() else block
        return request.override(system_message=SystemMessage(content=merged))

    # ---------------------------------------------------------------- hooks

    @override
    def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        return self._run_gate(state)

    @override
    async def abefore_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        # Archiving is local file IO and the interrupt is a control-flow signal;
        # neither benefits from an await, so the async path shares the sync body.
        return self._run_gate(state)

    @override
    def wrap_model_call(self, request: ModelRequest[ContextT], handler: Callable[..., Any]) -> Any:
        return handler(self._apply_injection(request))

    @override
    async def awrap_model_call(
        self, request: ModelRequest[ContextT], handler: Callable[..., Any]
    ) -> Any:
        return await handler(self._apply_injection(request))

    # ------------------------------------------------------------- utilities

    def search(self, needle: str) -> list[Record]:
        """Find archived messages by substring. Backs the post-hoc recovery path."""
        return self.archive.search(needle)

    def restore(self, message_id: str) -> Record:
        """Promote an archived message into the protected list."""
        rec = self.archive.get(message_id)
        if rec is None:
            raise GateError(f"memory-gate: no archived message with id {message_id!r}")
        self.archive.protect([rec])
        self.archive.mark_run({"reviewed": False, "reason": "manual_restore", "id": message_id})
        return rec

# Draft: langchain-ai/langchain — new issue

Filed under: `🚀: Feature request`. Title candidate:

> **SummarizationMiddleware: no public way to know that compaction is about to happen**

Verified against the installed release (`langchain 1.4.1`), not against `master`.

---

## The situation

I maintain a small middleware that wants to do one thing: run *before*
`SummarizationMiddleware` and, when compaction is about to discard the conversation,
hand the user a list of what is about to go and let them pin the parts that must
survive (https://github.com/2xIarge/memory-gate).

That is the same class of problem people are already reporting here — durable
constraints stated once in prose and then silently dropped:

- #36624 — plan drift and repeated re-planning from a lossy summary
- #38867 — "the caller sees no signal that data was destroyed"

To gate the operation you have to predict it. Today that means reimplementing it.

## What is public and what is not

Public and genuinely usable: `trigger`, `keep`, `token_counter`, `summary_prompt`.

But the two questions a gating middleware actually needs answered live behind
underscore methods:

**"Will it fire on this state?"**

```python
SummarizationMiddleware._should_summarize(messages, total_tokens)
SummarizationMiddleware._should_summarize_based_on_reported_tokens(messages, threshold)
SummarizationMiddleware._trigger_clauses
SummarizationMiddleware._get_profile_limits()
```

**"What exactly will it remove?"**

```python
SummarizationMiddleware._determine_cutoff_index(messages)
SummarizationMiddleware._partition_messages(messages, cutoff_index)
SummarizationMiddleware._find_safe_cutoff_point(...)
```

## The specific trap in `_should_summarize`

The `tokens` clause fires on the approximate count **or** the provider-reported total:

```python
if kind == "tokens":
    threshold_tokens = cast("int", value)
    # Trigger if total tokens exceed threshold OR reported tokens do
    if (
        total_tokens < threshold_tokens
        and not self._should_summarize_based_on_reported_tokens(
            messages, float(threshold_tokens)
        )
    ):
        clause_met = False
```

The first version of my gate did the obvious thing — call `count_tokens_approximately`
and compare against `trigger`. It looked correct, passed its own tests, and fired
later than the summarizer actually does. With a real model I watched a constraint
stated on turn 0 get compacted away with no review, because the model-reported total
crossed the threshold first. `fraction` has the same shape and additionally depends on
`_get_profile_limits()`.

A downstream reimplementation is therefore not "slightly off"; it disagrees with the
real predicate on exactly the inputs that matter.

## Ask

Two read-only methods on `SummarizationMiddleware`, no behaviour change:

```python
def would_summarize(self, messages: list[AnyMessage]) -> bool:
    """Same decision as the internal predicate, for middleware that run before us."""

def preview_cutoff(self, messages: list[AnyMessage]) -> int | None:
    """Where the cut would land, and which messages would be summarized vs kept."""
```

That is enough for gating, for observability ("compaction fired this turn, here is
what it took"), and for anything that wants to annotate or warn about the loss without
forking the predicate. I would take the second one even if the first is considered too
close to internals — the cutoff layout is what a reviewer needs to see.

## Why not the alternatives

- **Read the public `trigger`/`keep` and re-normalize myself.** That is what I do now.
  It duplicates `_normalize_trigger`, the clause/condition duality,
  `_should_summarize_based_on_reported_tokens` and `_get_profile_limits`, and it
  silently drifts on any patch release. My package exists only because I needed it; I
  would rather delete that code.
- **A `before_summarization` hook.** Cleanest, and more work. It would also give the
  community a place to attach budget warnings, loss reports and human gates without
  each of us reimplementing the trigger. If that is the direction you prefer, I would
  use it.
- **Subclass and override `_should_summarize`.** Then my middleware has to know that
  it is guarding *my* subclass, which defeats the point of it being generic.

Happy to open a PR for the two methods plus tests, if the shape is acceptable — and a
short "no" is more useful to me than a long maybe, because right now the only option I
have is to keep shipping the duplicate predicate.

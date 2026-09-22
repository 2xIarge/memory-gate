# memory-gate

**A fail-closed human gate in front of LangChain context compaction.**

`SummarizationMiddleware` replaces your conversation history with a summary an LLM
wrote. Whatever it decides is unimportant leaves the state — no exception, no log
line, no undo. `memory-gate` puts a person between "about to be summarized" and
"summarized".

```
pip install memory-gate
```

---

## The problem

Compaction is lossy, and the loss is silent. This is not hypothetical — it is what
people are reporting on the LangChain tracker:

> "The agent must then infer its plan from **a lossy LLM-generated summary** instead
> of reading structured state directly. This causes **plan drift and repeated
> re-planning** on long tasks."
> — [langchain#36624](https://github.com/langchain-ai/langchain/issues/36624), still open

> "The messages being deleted may be **the only record of a decision, instruction, or
> reasoning chain** … the caller sees **no signal that data was destroyed**."
> — [langchain#38867](https://github.com/langchain-ai/langchain/issues/38867) (that
> specific failure mode is now fixed; the shape of the complaint is not)

Structured state (`todos`, plans, config) can be made immune to compaction by
re-injecting it from `state` on every turn. **Everything else cannot.** A constraint
the user mentioned once, in prose, on turn 7 — *"our fiscal year starts April 1st"* —
has no schema and no home but the transcript. When the summary drops it, nothing
breaks. The agent just quietly starts answering "Q1" with January to March.

`memory-gate` protects exactly that class of information: the durable constraints
that only exist as dialogue.

## See it

`examples/fiscal_year_demo.py` runs offline with fake models. No API key.

```
$ python examples/fiscal_year_demo.py

# PART A -- SummarizationMiddleware alone
>>> 'fiscal year starts April' present anywhere in the request: False
>>> nothing raised, nothing logged. The agent will now answer 'Q1' as Jan-Mar.

# PART B -- MemoryGateMiddleware in front of it
====================================================================
memory-gate interrupted the run:
====================================================================
About to compact 6 messages (79 tokens of dialogue).
After compaction these survive only as a lossy summary.
Reply with refs to pin them permanently:

 [ 1] #6c9d9549…    14 tok  human  Analyse our quarterly revenue for 2026.
 [ 2] #6b5e3d87…    16 tok  ai     Sure. Which fiscal calendar should I use?
 [ 3] #ac5b5611…    29 tok  human  By the way, our fiscal year starts April 1s…
 [ 4] #4630d18a…     7 tok  human  run query 0

  keep 1,3,5   pin the listed refs
  keep all     pin everything listed
  confirm      compact without pinning

you -> 'keep 3'

>>> 'fiscal year starts April' in the message window: False  (compaction destroyed it)
>>> present in the request via injection: True
>>> restored specifically by memory-gate: True
archive on disk: {'archived': 16, 'archived_tokens': 317, 'protected': 1}
```

Note the last three lines. The original message really *was* destroyed — it is gone
from the message window. It reaches the model again only because the gate put it
back.

## Quickstart

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from memory_gate import MemoryGateMiddleware

summarizer = SummarizationMiddleware(model="openai:gpt-4o-mini",
                                     trigger=("tokens", 24000), keep=("messages", 6))

agent = create_agent(
    model,
    tools=[run_sql, export_chart],
    middleware=[
        MemoryGateMiddleware(               # <- MUST be first
            summarization=summarizer,
            archive_dir=".memory-gate",
            protect_patterns=(r"fiscal year", r"do not touch staging"),
        ),
        summarizer,
    ],
    checkpointer=checkpointer,              # required: interrupt() needs one
)
```

Ordering is not cosmetic. `before_model` hooks become their own graph nodes chained
in list order, and `wrap_model_call` composes *first = outermost*; being first means
the gate sees every message before compaction can delete it, and gets the last word
before the model is called. Verified against the installed release, not just the docs.

## How it works

**1. Archive first, unconditionally.** Every message that falls outside the keep
window is appended to `archive.jsonl` before any review or compaction happens. This
is idempotent and never asks permission. Over-archiving costs disk; under-archiving
loses data permanently.

**2. Ask only when compaction is actually imminent and the loss looks expensive.**
The gate reads the guarded middleware's own `trigger` rather than guessing. Without
this alignment a gate interrupts on nearly every turn — in an early build it fired
13 times in 16 turns, which is the fastest route to being uninstalled. With it, the
same run interrupts twice. It also never asks about a candidate set it already
showed you.

**3. Pin, then re-inject on every call.** Pinned text lands in `keep.jsonl` and is
merged into the system message on each subsequent model call, so later rounds of
compaction cannot take it back.

## Guarantees

Every failure path is chosen so that a broken gate is visible, never quietly absent.

| Situation | Behaviour |
|---|---|
| Archive cannot be written | Raises. Compaction does not happen. |
| Reply cannot be parsed | Keeps **everything** listed. Never guesses. |
| No `summarization` and no `trigger` given | `ValueError` at construction. A gate that can never fire is worse than no gate. |
| Candidate set changed during resume | Raises. Refuses to compact a list you did not approve. |
| Review set identical to one already answered | Skips the ask. Decision is already on disk. |
| Compaction summary reaches the review list | Excluded by `lc_source=summarization`. You are never asked to preserve a paraphrase. |

The last table row matters more than it looks. Ask a human "do you want to keep this
LLM-written summary?" and you have asked a question with no good answer.

## What this is not

- **Not a compaction algorithm.** It does not summarise better. It decides what must
  not be summarised, then lets the host middleware do its job.
- **Not a token saver.** Pinning costs tokens. This buys correctness.
- **Not a way to stop the state update.** Compaction still rewrites history; the gate
  restores what you pinned from an on-disk archive. Anything you did not pin is
  recoverable from `archive.jsonl`, but not automatically.
- **Not for tool results.** They are reproducible and are the bulk of the tokens, so
  they are archived but never put in front of you. That keeps the review list short
  enough to actually read.
- **Not a general memory system.** For durable knowledge across sessions, use
  `claude-mem` or similar. This guards one session against one specific operation.

## Recovery

```python
gate.search("April")        # substring search across everything ever archived
gate.restore("ac5b5611-…")  # promote a forgotten message into the pinned set
gate.archive.stats()        # {'archived': 16, 'protected': 1, ...}
```

Judging what matters *before* the loss is the hard part, and people get it wrong.
Every gate run is recorded to `runs.jsonl`, so you can audit what was offered, what
you approved, and what was dropped without asking.

## Requirements

Python >= 3.10, `langchain >= 1.0`, `langgraph >= 1.0`, and a checkpointer (required
by `interrupt()`). Developed and verified against `langchain 1.4.1` /
`langgraph 1.2.11`.

It depends on `before_model` being a separate graph node. That is what keeps
`interrupt()` out of reach of `ModelRetryMiddleware` / `ModelFallbackMiddleware`,
which historically swallowed interrupts
([langchain#38837](https://github.com/langchain-ai/langchain/issues/38837)). If that
composition changes, so does this package — hence the explicit test for it.

## Run the checks

```
python scripts/verify.py      # 45 staged assertions, offline
python examples/fiscal_year_demo.py
```

## License

Apache-2.0.

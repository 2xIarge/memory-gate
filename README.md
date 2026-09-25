# memory-gate

**A fail-closed human gate in front of LangChain context compaction.**

`SummarizationMiddleware` replaces your conversation history with a summary an LLM
wrote. Whatever it decides is unimportant leaves the state — no exception, no log
line, no undo. `memory-gate` puts a person between "about to be summarized" and
"summarized".

```
pip install git+https://github.com/2xIarge/memory-gate.git
```

Not on PyPI yet — `pip install memory-gate` returns 404 today. The name is unclaimed
and publishing is pending; until then install from git, or clone and run
`pip install -e .`.

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
Compaction will drop 6 messages. 4 of them are dialogue you can keep (57 tokens).
The rest survive only as a lossy summary. Nothing is deleted -- all of it stays in the archive.

 [   2]*   29 tok  By the way, our fiscal year starts April 1st, so Q1 is Apr-Jun. All pri…
 [   1]    14 tok  Analyse our quarterly revenue for 2026.
 [   3]     7 tok  run query 0
 [   4]     7 tok  run query 1

 * matched a protect pattern: 1 of 4 look like a standing rule.
   The patterns miss some real rules, so an unmarked line is not a line you can afford to lose.

 2 further messages are being dropped and are not listed (tool results, and assistant replies unless pin_roles is widened).

  keep all     pin all 4, budget permitting   <- the usual answer
  keep 1,2,3  pin only these refs
  confirm      compact and pin nothing new

you -> 'keep 2'   (pinning the fiscal-year constraint)

>>> 'fiscal year starts April' in the message window: False  (compaction destroyed it)
>>> present in the request via injection: True
>>> restored specifically by memory-gate: True
archive on disk: {'archived': 16, 'archived_tokens': 317, 'protected': 1, 'protected_tokens': 29}
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

`protect_patterns` **replaces** the defaults rather than adding to them. To keep the
measured set and extend it, pass `DEFAULT_PROTECT_PATTERNS + (r"your marker",)`.

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

## Measured

One model (local Qwen through llama.cpp), one 57-turn transcript with 10 constraints
planted among 40 unrelated topics, trigger 2 500 tokens / keep 6 messages. Every
configuration below is a **single** run — read the noise floor before comparing two
numbers in this section.

| run | triage the reviewer was given | reviewer's answer | baseline alone | with the gate |
|---|---|---|---|---|
| C1 | imperative-only set that never saw the planted lines — **coverage 4/10** | `keep <flagged>` | 7/10 | **7/10** |
| C2 | shipped default set — **coverage 10/10**, see caveat 3 | `keep all` | not run | 8/10 |
| R2 | shipped default set | `keep <flagged>` | not run | 10/10 |
| ~~old~~ | **answer keywords = oracle** | `keep <flagged>` | 8/10 | ~~10/10~~ superseded, see caveat 2 |

What the runs establish:

- **Under `keep <flagged>`, the benefit is bounded by triage coverage.** In C1 the four
  flagged sentences (`营收口径`, `数据环境`, `货币单位`, `交付节奏`) all survived in the
  baseline as well, and every constraint lost in either arm was one of the six the set
  missed. Coverage 4/10, gain 0.
- **Under `keep all`, the binding constraint is the budget.** C2 pinned 58 messages
  (549 tokens) against a resolved budget of 200, so only the newest 23 records
  (198 tokens) were re-sent; the two constraints it lost were pinned but never injected.
  72% of the pinned tokens were not constraints. `render_review_text` prints the
  overflow in tokens for exactly this reason.
- **Blanket pinning and a proportional budget are not a matched pair.** Ranking pins --
  explicit refs above flagged above blanket `keep all` -- would address both findings,
  and is not implemented.

Caveats, in the order that changes the reading:

1. **n = 1 per configuration, and the baseline itself is noisy.** The same Part A
   transcript scored 7/10 on one run and 8/10 on another. A difference of one is inside
   the noise; 7 vs 10 is not.
2. **The 8/10 → 10/10 result this README used to lead with is withdrawn.** That run
   passed `protect_patterns` built from the planted constraints' own answer keywords
   (`财年`, `net`, `_v3`, `Lena`, …) plus four extra markers, and then answered each
   review with the flagged refs — the reviewer held the answer key, so the flagged items
   *were* the graded answers. What it does still support is narrower: anything that
   *is* injected survives compaction verbatim — the plumbing, the idempotent archive and
   the budget all had to work for that.
3. **The shipped default set is not blind to this transcript.** Several of its
   declarative markers (`按这个来`, `习惯了`, `顺手记`, `只用`, `都这么`, `记得`) were written
   after reading the experiment's own missed-constraint list, so C2's and R2's 10/10
   coverage is partly self-fulfilling; `verify_real_model.py --patterns original` is the
   blind control. Every run header prints which set was used, its overlap with the
   planted sentences, and its coverage.
4. **A 2 500-token trigger exaggerates the budget mismatch.** 36% of pinned tokens
   reached the model in C2; the same 2% rule against a 160K window covers roughly 87% of
   a session's user turns. The mechanism is real, the severity in production is milder.
5. **Loss depends on how aggressively you compact.** Loosen `keep` and the baseline
   holds 10/10 on its own; if you control that setting, changing it is cheaper than
   adding a gate. This package is for configurations you do not control — constrained
   context windows, shared agents, tool-heavy sessions.
6. **One review per compaction, not nagging** — C2 asked 5 times and compacted 5 times
   over 57 turns. Pinning does raise reported usage, which is the runaway the budget
   exists to cap (measured: 2 cycles became 31 with a budget near half the trigger).
7. `pin_roles` defaults to `("human",)`. The assistant restates your rules in its own
   words; pinning those too spent the same budget on redundant text and roughly halved
   how many distinct constraints fitted.
8. `protect_budget_tokens` defaults to 2% of the trigger (clamped to [200, 4000])
   rather than a constant, and warns when an explicit value is over 10% of it — because
   pinned text is re-sent and re-counted every call, a budget comparable to the trigger
   re-arms the compaction it just survived. Caveat 4 is the cost of that being too small.

## Which lines look like rules

The `*` marks come from `DEFAULT_PROTECT_PATTERNS`, and their accuracy is a product
property because it is the only triage a reviewer gets. Scored by
`scripts/score_patterns.py` on corpora the patterns were not tuned against:

| pattern set | held-out recall | held-out precision | flagged share of a real transcript |
|---|---|---|---|
| imperative only (the original) | 29% / 58% | 67% / 88% | 16% |
| current default | 93% / 83% | 93% / 100% | 17% |

The gap is structural, not a tuning problem: the original set was entirely imperative
markers (`must`, `never`, `必须`, `一律`), and people state conventions as *facts* far
more often than as commands — "our fiscal year starts April 1st", "对接人是 Lena". The
first version of this fix was tuned by reading the same transcript it was then scored
on; it looked 100% better on that transcript and *worse* on held-out lines. Hence the
split corpora, and the failing control group kept in the script.

A pattern list has a ceiling. `周报发给 Kevin，抄送整个组` is a durable rule with no
marker word in it, and no regex reaches it. That is why `keep all` is the default
action and the flags are a reading aid, never a filter.

## How big the list gets

`scripts/render_at_scale.py` renders a production-scale review (200K window,
compaction at 80%) through the real renderer. List length turns out to be driven by
*how many turns the user took*, not by token volume, since the mean human turn
measured 9.4 tokens:

| workload | user turns to fill 160K | rendered lines |
|---|---|---|
| tool-heavy agent | 10 | 22 |
| light tools | 48 | 53 |
| long chat, no tools | 390 | 54 (was 403 before the cap) |

The uncapped chat case was 403 lines / 16,497 characters, which nobody reads. So
`render_review_text` shows 40 rows — rules first, then oldest — while the payload keeps
every candidate. That distinction matters: truncating by the *pin budget* instead would
drop the oldest messages, which is exactly where a rule stated once at the start of a
long session lives.

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
  they are archived but never put in front of you. This is necessary but not
  sufficient: at chat scale the user's own turns alone still measured 403 lines, which
  is why the renderer caps what it shows.
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
python scripts/verify.py           # 47 staged assertions, offline
python scripts/score_patterns.py   # what the `*` marks actually catch
python scripts/render_at_scale.py  # the review block at a 200K window
python examples/fiscal_year_demo.py
python -m pytest                   # 50 tests
```

## License

Apache-2.0.

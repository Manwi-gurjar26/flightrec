# flightrec

**A flight recorder for LLM agents.** Record every step of an agent run, replay it
deterministically from any step, and diff two runs to find exactly where they diverged.

---

> ### The number
>
> **Replay reproduces the recorded step sequence 100% of the time at temperature 0 — up from 0% before pinning tool results, timestamps and retry jitter.**
>
> *Status: not yet measured. This README was written before the code, deliberately.
> Every number below is a **target with a defined measurement procedure** (see
> [Measurement](#measurement)). Each one is replaced with a measured value, or an
> honest worse one, when step 9 of the build lands. Nothing here is quoted on a
> resume until it comes out of `flightrec bench`.*

---

## The problem

Agents fail in the middle, not at the end.

The seventh tool call returns an empty list. The agent doesn't crash — it invents a
recovery that looks plausible and is wrong. Four steps later you get an answer that is
confidently incorrect, and the only artifact you have is 4,000 lines of JSON in a
terminal scrollback.

Three things are broken about debugging that:

1. **Print-statement debugging stops working past about three steps.** The state you
   need to see is a tree, not a line.
2. **Failures can't be reproduced.** Every run is different, so "run it again and watch"
   is not a debugging strategy.
3. **Nobody can answer "why did this run cost 8x more than the last one?"** — because
   nobody is lining the two runs up against each other.

`flightrec` is the readable version: a timeline of every step showing what the agent was
reasoning about, which tool it called, what came back, what it cost, and the precise
moment things went wrong — plus the ability to *go back to that moment and change one
thing*.

## What it does

- **Trace.** A small SDK (one decorator, one context manager) wraps model calls and tool
  calls and emits spans following the OpenTelemetry GenAI semantic conventions — a real
  tracing format, not a private invention.
- **Store.** A FastAPI collector writes span trees to SQLite. A run is a tree, so the
  schema carries parent references.
- **See.** A step timeline with expandable detail, prompt and response side by side, a
  token/cost bar per step, and red markers on errors and retries.
- **Replay.** Re-run from step N with every earlier step served from the recording. Change
  one prompt and see the effect without paying for the first six steps again.
- **Diff.** Put two runs side by side, align their steps properly, and point at the first
  genuine divergence.

```bash
flightrec demo --seed 0 --db flightrec.db   # record a run that goes wrong
flightrec serve --db flightrec.db           # open http://127.0.0.1:8000

flightrec replay <run_id>                   # reproduce it, step for step
flightrec replay <run_id> --from-step 4     # rerun live from step 4 onward
flightrec replay <run_id> --from-step 4 --strict   # stop there instead
```

The UI is server-rendered Jinja2 with **no JavaScript and no external requests** —
expandable steps are native `<details>` elements. The collector is a local tool that
has to work offline, and the alternative was vendoring a JS library to reimplement a
browser primitive.

## Why this was hard

*(The three sections below are the point of the project. They get filled in with what
actually happened — what was tried first, why it failed, what replaced it — as each piece
lands. Placeholders are marked TODO and are not allowed to survive to v1.0.)*

### 1. Deterministic replay: every source of variation has to be pinned

A replay that isn't bit-identical isn't a replay, it's a second run wearing a costume —
and the failure is silent, which is the worst property a debugging tool can have.

The sources of non-determinism, each of which has to be independently pinned:

| Source | How it leaks | Pinned by |
|---|---|---|
| Sampling | temperature / top_p / seed | recorded in the span, forced on replay |
| Tool results | live call returns different data | served from the recording, never re-executed |
| System clock | `datetime.now()` inside a prompt | virtual clock seeded from the recording |
| Retry/backoff | jitter changes the call sequence | seeded RNG, recorded seed |
| Iteration order | `set` / `dict` ordering across processes | `PYTHONHASHSEED` recorded and forced |

**The one that actually bit us, and it was not on the list: clock resolution.**

The first recorded trajectory came back with the calculator call *ahead of the
page fetch that produced its operands*. Nothing was wrong with the agent. The
trajectory was sorted by `(start_time, span_id)`, `time.time()` on Windows
resolves to about 15.6ms, and agent steps finish in microseconds — so eleven
spans shared three distinct timestamps and the tiebreaker was a random UUID.

Two fixes, because one was not enough:

1. `SystemClock` anchors to the wall clock once and advances with
   `perf_counter()`. Sub-microsecond resolution, monotonic, still reads as epoch
   seconds. This also stopped every short span reporting a duration of exactly
   zero.
2. More importantly, **ordering no longer depends on time at all.** Each span
   carries a monotonic `sequence` counter assigned at start, and that is the
   ordering authority. Timestamps are for display and duration only.

Fix 1 alone would have hidden the bug rather than fixed it — the clock was
precise enough afterwards that time-based sorting appeared to work on this
machine, and would have broken again on a coarser platform or across processes.
That is also why the regression test injects a deliberately coarse clock instead
of using the real one; with the real clock it passed no matter which sort was in
use, which is a test that proves nothing.

**The second thing that bit us: "bit-identical" is two different claims, and only
one of them is achievable.**

A replay cannot recover the recording's span IDs or its wall clock — the
recording used real UUIDs and a real clock, and nothing brings those back. So
the guarantee is deliberately split, and both halves are tested:

* **replay vs. replay is bit-identical**, span IDs and timestamps included,
  because the ID generator and clock are seeded from the recording;
* **replay vs. recording is trajectory-identical** — same steps, inputs, outputs
  and statuses.

Claiming the stronger version for both would have been the silent kind of lie
this tool exists to catch, so `step_signature()` states exactly what is compared
and why span IDs are excluded.

Three more things that had to be pinned, none of which were on the original list:

| Source | How it leaked | Pinned by |
|---|---|---|
| Tool-result lookup key | keying the oracle by `(name, arguments)` serves the first `fetch_page` result to a later retry of the same URL, silently deleting the retry | serving in recorded **order**, with name and arguments checked afterwards — a disagreement is a loud `ReplayMismatch`, not a lookup miss |
| Fault configuration | live steps past the edit point ran in a fault-free world, so they were not a continuation of the recorded run | fault rates recorded on the root span and restored on replay |
| Span ID space | seeding IDs from the *agent's* seed made every replay of every same-seeded recording share span IDs — and storage writes are `INSERT OR REPLACE`, so two replays would have silently overwritten each other's steps rather than colliding | ID seed derived from the recording plus the edit, so replays of one recording stay identical and different ones stay apart |

The last one is the one worth remembering: it would never have surfaced as a
wrong replay. It would have surfaced weeks later as a run in the database with
somebody else's steps in it.

### 2. The forward-execution problem, which has no obvious right answer

Once you edit step 3, every step after it is invalid — the recording is no longer a
description of what this modified run would do.

**Decision: re-execute forward with live calls, and mark every step after the edit point
as `divergent` in the UI.** A `--strict` flag stops at the edit point instead of guessing.

Rationale: the whole reason to edit step 3 is to see what happens *afterwards*; stopping
there answers nothing. But a user must never be able to mistake a live-executed step for a
recorded one, so divergence is a visible property of every step, not a footnote.

What using it changed: **"divergent" was not a fine enough distinction.** The
first version marked every span in a replay as `replayed` and everything past
the edit point as `divergent`, which reads as "these steps came from the
recording, those didn't" — and that is wrong. A model call is *re-executed* even
in a perfectly faithful replay; only tool results are served from the recording.
Labelling a re-executed model call as recorded data misrepresents provenance on
exactly the steps a user is trying to reason about. There are now three states,
and the timeline shows which is which:

| Label | Meaning |
|---|---|
| `recorded` | the tool result was read back out of the recording |
| `re-run` | re-executed, and expected to match — every model call, always |
| `live` | past the edit point; the recording no longer applies |

TODO: the cost implications of forward re-execution, measured rather than
asserted, in step 9.

### 3. Run diffing is sequence alignment, not `zip()`

Two runs have different numbers of steps. Run A retried a tool twice; run B didn't. Pairing
them index-by-index means every step after the first insertion is compared against the
wrong partner, and the tool reports garbage divergence from that point on.

This is a sequence alignment problem. It needs:

- a **similarity function** over steps — tool name, arguments, outcome class — not string
  equality of the whole span;
- an **alignment algorithm** (Needleman–Wunsch, affine gap penalties) to line the two
  sequences up before any comparison happens;
- only then, a **first genuine divergence** to point at.

TODO: the gap penalty tuning, and the cases where alignment still gets it wrong.

## Architecture

```
  your agent code
        │  @trace / with span(...)
        ▼
  flightrec SDK ──── spans (OTel GenAI conventions) ────┐
        │                                               │
        │ replay: tool results served from recording    │
        ▼                                               ▼
  replay engine  ◄──── run trees ────  collector (FastAPI) ──► SQLite
        │                                               │
        └──────────► diff (Needleman–Wunsch) ◄──────────┘
                             │
                             ▼
                   timeline UI (Jinja2 + HTMX)
```

## Install

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -e ".[server,dev]"
pytest
```

## Measurement

Every number in this README comes from `flightrec bench`, which is committed and
reproducible. No number is reported without a baseline to compare it against.

| Metric | How it is measured | Baseline | Target |
|---|---|---|---|
| **Replay fidelity** | Fraction of N recorded runs whose replayed step sequence is identical to the recording, at temperature 0 | Same measurement with pinning disabled | 100% vs. ~0% |
| **Divergence localization** | Of M runs with a synthetically injected divergence at a known step, the fraction where diff reports that exact step | Index-by-index `zip()` pairing | alignment ≫ zip |
| **Replay cost saving** | Provider tokens spent replaying from step N vs. re-running from step 0 | Full re-run | — |
| **Overhead** | Wall-clock and token overhead the SDK adds to an uninstrumented run | Uninstrumented agent | < 5% wall clock |

The demo agent used for all measurements is committed (`examples/research_agent.py`) and
runs against a deterministic stub model by default, so anyone can reproduce these numbers
without an API key.

## Build status

- [x] 1. Repo skeleton + README contract
- [x] 2. Span model + tracing SDK
- [x] 3. Demo agent that genuinely fails
- [x] 4. Collector + SQLite storage
- [x] 5. Timeline UI
- [x] 6. Token / cost rollups
- [x] 7. Deterministic replay
- [ ] 8. Run diff with sequence alignment
- [ ] 9. Measurement harness → real numbers in this README

## What I would do with two more weeks

- **Flaky-step report.** Run the same task 50 times, cluster the trajectories, and rank
  steps by outcome variance. The high-variance steps are the ones the prompt is handling
  by luck.
- **Score the replay, not just reproduce it.** Attach a rubric-based scorer to each
  replayed run so "I changed step 3's prompt" produces a quality delta, not just a
  different transcript.
- **Trajectory-level regression gate in CI.** Block a PR when the agent's step sequence on
  a fixed task set changes in a way that costs more or scores worse.

## Licence

MIT.

# flightrec

**A flight recorder for LLM agents.** Record every step of an agent run, replay it
deterministically from any step, and diff two runs to find exactly where they diverged.

---

> ### The number
>
> **Replay reproduces the recorded step sequence in 40 runs out of 40. Re-running
> the same task without the recording reproduces it in 8 of 40.**
>
> *Status: measured. `flightrec bench` produces this and every other number
> below, in about a minute, with no API key and no network.*
>
> *This README was written before the code, deliberately, with each number as a
> target and a defined measurement procedure. Two targets were met — the second
> only after the benchmark was made harsh enough to break the thing it was
> measuring — one had no target, and one was missed by a factor of about twenty.
> See [Measurement](#measurement), where the miss is written up rather than
> quietly dropped.*

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

flightrec diff <run_a> <run_b>              # align them, find the divergence
flightrec diff <run_a> <run_b> --by-index   # the naive pairing, for comparison

flightrec bench                             # reproduce every number below
```

The UI is server-rendered Jinja2 with **no JavaScript and no external requests** —
expandable steps are native `<details>` elements. The collector is a local tool that
has to work offline, and the alternative was vendoring a JS library to reimplement a
browser primitive.

## Why this was hard

*(The three sections below are the point of the project: what was tried first, why
it failed, what replaced it. They are written after the fact and they are not a
success story — several of the bugs recorded here were found by measuring
something the tests already said was fine, and section 3 ends with two failures
that are still failures.)*

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
| `recorded` | read back out of the recording, and billed nothing |
| `live` | past the edit point; really executed, because the recording no longer applies |
| `stopped` | where `--strict` gave up rather than guess — not a step that ran |

**What the cost measurement then found, which changed the design.** The first
implementation served tool results from the recording and re-executed every
model call. It replayed *correctly* — trajectories matched, every test passed —
and it quietly failed the part of the promise that makes the feature worth
having: "without paying for the first six steps again". Re-executing the model
calls means paying for them. Nothing caught it until step 9 tried to measure a
saving and found none, because a replay that is faithful and expensive looks
exactly like a replay that is faithful and cheap from the outside.

Steps before the edit point now serve model responses too, and the served
response is checked against the prompt the replay actually produced — serving a
recorded answer to a question that was never asked would be the same silent
class of bug one level down.

### 3. Run diffing is sequence alignment, not `zip()`

Two runs have different numbers of steps. Run A retried a tool twice; run B didn't. Pairing
them index-by-index means every step after the first insertion is compared against the
wrong partner, and the tool reports garbage divergence from that point on.

This is a sequence alignment problem. It needs:

- a **similarity function** over steps — tool name, arguments, outcome class — not string
  equality of the whole span;
- an **alignment algorithm** (Needleman–Wunsch, affine gap penalties) to line the two
  sequences up before any comparison happens;
- a **second pass to recover reorderings**, because the alignment above is
  monotonic and a reordering is not — see [below](#where-the-alignment-gets-it-wrong);
- only then, a **first genuine divergence** to point at.

**The gap-open penalty is a tie-breaker, not a force, and tuning it as a force
is a bug.** It was set to -0.6 first. With that value the aligner preferred
pairing a *tool* step against a *chat* step — similarity 0.0, nothing in common
whatsoever — over opening a second gap. The ceiling is set by the similarity
function itself: its smallest component is 0.2, which is 0.4 in score terms, so
a gap-open above that outbids real evidence. It is now -0.25, and a test asserts
both the pairing and the bound.

**Affine gaps buy less than the textbook suggests.** The first test written for
them asserted that an inserted three-step block stays contiguous — which it
does, under affine *and* under every linear penalty tried. The steps around the
block match perfectly, so nothing is gained by splitting it and no penalty
scheme splits it. That test proved nothing, exactly like the clock test in
section 1 would have with a real clock. The case where affine genuinely decides
the outcome is narrower: when several candidate partners score *identically*,
one gap of two and two gaps of one are exactly tied under a linear penalty and
the winner comes down to evaluation order. Affine makes "one event produced one
block" the principled answer rather than the lucky one. That is worth having,
and it is a smaller claim than "linear shreds retry blocks".

**The corpus could not demonstrate any of this until the agent was changed to
allow it.** Every one of the 40 seeds originally produced a trajectory of
exactly 11 steps, because the demo agent's policy was fixed — search, fetch,
search, fetch, calculate, answer. Faults changed *what* each step returned,
never *how many* there were. With no length difference, alignment and `zip()`
produce identical output, so the diff's entire reason for existing was
untestable against real recordings and the "alignment ≫ zip" claim would have
measured as "alignment == zip".

The fix was to give the agent something real agents do: when a guessed URL 404s,
it tries a second scheme before giving up. Runs are now 11, 13 or 15 steps
depending on how many searches came back empty. Both guesses are wrong on
purpose — letting the second one succeed would have been a nicer story and would
have deleted the demo's most important failure, an empty search leading to
invented data.

That is what the divergence-localization measurement runs against: a real
recorded recovery block spliced into a copy of a run, plus one later step's
output changed. Alignment pairs the changed step with its counterpart in 40 of
40 mutants; index pairing manages 0, because the insertion shifts everything
after it. Note what the baseline does there — it *does* report a difference at
the right index, about the wrong step. Scoring "reported something at index k"
would have handed it a pass for being confidently wrong, which is the failure
mode this project is about.

### Where the alignment gets it wrong

The first version of that benchmark scored 100%, which was a fact about the
benchmark rather than about the aligner: it only ever injected one insertion and
one changed result. Six classes replaced it — insertion, deletion, insertion
*and* deletion, two changes at once, reordering, and one tool substituted for
another — scored separately, because an average over classes hides a total
failure behind five easy passes. (Four harder ones came later; see
[below](#making-the-corpus-harder-again-and-the-metric-that-could-not-see).)

Two of them broke it. Both are fixed; what follows is what they were, because
the second fix is the more interesting half of this section.

**1. Reordering, which no amount of tuning would fix.** Needleman–Wunsch produces
a *monotonic* correspondence: step order is preserved on both sides. A
reordering is by definition non-monotonic, so it could not be represented at all.
Localization dropped to 72%, and every failure involved a step whose
correspondence crossed another's. This was never a penalty-tuning problem —
run `flightrec bench` against the first pass alone and the failure comes
straight back, which a test pins.

**2. An insertion and a deletion that cancel out in length, which was the more
embarrassing one.** Two steps added and two removed leaves the run the same
length. The aligner then discovers that pairing every step positionally costs a
handful of substitutions, while representing the truth costs two gap blocks at
−1.45 each — so it took the cheap option and **degenerated into exactly the
index-by-index pairing this whole module exists to beat**. On the failing cases
the diff contained no gaps at all: five matches, six changes, every step paired
with its positional neighbour.

The second one is worth dwelling on, because localization scored 100% for that
class throughout. The changed step sits after the deletion, where the net offset
is back to zero, so the headline metric was satisfied by an alignment that was
wrong about nine steps out of eleven. That is why the benchmark reports a second,
harsher number — the fraction of *all* surviving steps paired with their true
counterpart. Localization said 100%; pairing said 91.9%, and the gap between
those two numbers is what would have shipped unnoticed.

### The fix: a second pass, not a better score function

Move recovery runs after the alignment and only touches pairings the first pass
was not confident about. Gaps and weak substitutions go back into a pool and get
re-matched, strongest candidate first, by similarity above 0.9 — which is what
catches a step that was moved *and* edited, since its content differs and only a
function that ignores output can still recognise it. Whatever pairs remain out of
order relative to the longest consistent backbone are the ones that moved.

(Identical steps originally got their own earlier pass, on the reasoning that
exactness is the strongest evidence available. That turned out to be wrong in a
way worth reading: see [below](#position-has-to-break-the-tie-that-content-cannot).)

It could not just examine gap blocks the way a line differ does. A reordering
preserves length, so in the common case the first pass emits **no gaps at all** —
it pairs everything positionally through a chain of substitutions, the same
degeneracy as failure 2. Half the reorder cases came out as `rem rem … ins ins`
and half as six substitutions and nothing else, and a fix that only understood
the first shape would have looked like it worked.

Three bugs in the pass itself, all found by re-running the benchmark rather than
by thinking:

| Symptom | Cause |
|---|---|
| deletions fell 100% → 70% | the pass walked left steps in order, so a *deleted* step reached the changed step's counterpart first and paired with it — losing the deletion and the edit together. Now every candidate is scored and the strongest pair is taken first. |
| insertions fell 100% → 97.5% | a correctly-paired changed step was treated as "weak" and thrown back into the pool, where an exact-match candidate took its counterpart. Confident changed pairs are now protected. |
| reorderings stuck at 97.5% | protection used the same 0.9 threshold as matching, so a step paired with its *positional* neighbour at 0.955 was protected and never re-examined. Two decisions, two constants: protection now requires total certainty. |

`identical` also had to learn about order — every column of a pure reordering is a
`MATCH`, so a diff that only checked content reported two runs that did the same
things in a different sequence as the same run.

### The last 0.6%, which was not an alignment problem at all

That left pairing accuracy at 99.4% on insertion-plus-deletion and 99.7% on
reorderings — two mutants out of forty, and the temptation was to tune something.
The cause was in neither pass.

**A failed step records an output of `None`, so every failure looked alike.**
`_classify` compared the tool name, the output and the status, and concluded that
two fetches of two *different* URLs that both 404'd were the same step with
reworded arguments — `cosmetic`. It never compared the error message. So the
diff was quietly telling users that a run which failed to fetch
`seattle-climate` and a run which failed to fetch `portland-annual` differed only
in phrasing, **on precisely the steps where those runs went wrong**.

The pairing damage was a side effect: a column called cosmetic is treated as
settled, so the move pass never re-examined it, and a step that had an exact
match waiting three positions away stayed paired with a stranger. Comparing the
message fixed both numbers at once — localization and pairing reached 100% on all
six classes then in play — but the classification bug was the more serious half,
and it would have survived any amount of work on the aligner.

`step_signature` had the same hole and got the same fix, so `MATCH` cannot claim
two steps are identical when they failed for different reasons.

### Making the corpus harder again, and the metric that could not see

Six classes with every number at 100% is a benchmark that has stopped
discriminating. Four more were added — an insertion butted directly against a
deletion, a step duplicated verbatim elsewhere, several edits compounded into one
run, and a run where most arguments are reworded to no effect and exactly one
result genuinely changes.

The first thing that fell out was not about the aligner. **All three existing
metrics were blind to an entire class of wrong answer.** `localized` looks only
at the changed step; `pairing` scores only steps that *survived*. Neither can see
a diff that pairs two completely unrelated steps and calls it a change, where the
truth is "one removed, one unrelated one added". So a fourth metric now asks
whether added and removed steps are reported as added and removed — and
`adjacent-edit` scores **0%** on it while scoring 100% on everything else.

**That one is a genuine trade-off, not a bug.** The aligner pairs the injected
steps with the deleted ones because two gap blocks cost −2.9 and two weak
substitutions cost −1.2. It could be made to prefer gapping, except that
"one step replaced by an unrelated step" is *locally identical* to a tool
substitution — same shape, same similarity of about 0.2 — and the `substitute`
class exists because pairing those is what the diff is supposed to do. The two
expectations contradict each other. Tuning until both pass is not possible;
picking one and calling the other a limitation is honest.

Two more corrections, both to the benchmark rather than the code:

- **`adjacent-edit` was vacuous when first written.** It spliced in a *copy* of
  real steps, which matched their neighbours, so the aligner slipped a match
  between the two gaps and the adjacency it was built to force never happened. It
  scored 100% while measuring nothing — the same failure as the affine-gap test
  in section 3, made twice. The injected steps are now unmatchable by
  construction, and a test asserts it.
- **The structure metric reported failures that were its own fault.** A step
  injected as a copy of one deleted elsewhere *is* a move and the diff was right
  to say so; a step duplicated verbatim leaves two identical steps and which twin
  is "the extra one" has no answer. Both were being marked wrong. Ground truth
  now scores only what is decidable and returns "no answer" otherwise, which is
  why `duplicate` shows a blank structure column rather than a bad score.

### Position has to break the tie that content cannot

`duplicate` and `compound` were the two classes still failing, and they turned
out to share a cause worth stating on its own.

The move pass rescued identical steps in a first pass, with no threshold,
before considering anything else — exactness looked like the strongest possible
evidence. It is not, once a run contains a copy of one of its own steps.
`compound` splices in a recovery block that happens to contain a step identical
to one later in the run, and then edits that later step. Now the copy is an
*exact* match for the left step while the real counterpart differs by its output.
Both score similarity 1.0, because similarity ignores output on purpose — so
exactness cannot separate them, and the "obviously stronger" evidence sent the
pairing six positions away from where the run said it belonged.

The fix is a single ranking: similarity first, then distance from where the
step's *settled neighbours* say it belongs. Raw index distance would not do it
either — after two steps are inserted everything later is two out, and the right
candidate can be the further one. Anchoring on the nearest confidently-paired
neighbour carries the local offset. Both classes went to 100%.

Two of the remaining `duplicate` failures were the benchmark's fault again. The
mutation duplicated a step and then sometimes edited one of the twins, which
leaves the run holding both the old and the new version of one step — the diff
can pair the unedited twin and call the edited one an insertion, and that
describes the same pair of runs. A mutation that tests two things at once cannot
be scored on either, so the duplicated step is now barred from also being the
edited one. Scoring was fixed to match: two mutant steps with identical
signatures are the same answer, and picking between them is a coin toss.

### Ground truth was wrong three times, in the same way each time

`insert+delete` sat at 78% on `structure` and the assumption — mine, stated
twice before checking — was that it was the same replacement-versus-substitution
trade-off as `adjacent-edit`. It was not. It was the ground truth being
over-specified, for the third time:

| What the mutation did | What the diff said | Both true? |
|---|---|---|
| copied a step to where another was deleted | "it moved" | yes |
| duplicated a step verbatim | paired the other twin | yes — they are identical |
| deleted a `fetch_page`, added a different `fetch_page` | "one call, changed URL" | yes |

Each time the benchmark demanded one description where two are equally correct,
and each time the diff was marked wrong for being right. So a removed step is
now scored only when no added step could be taken for it — same kind and same
tool name is enough to be confusable. That is a structural test, deliberately
not the diff's own 0.9 threshold, which would be scoring the diff against its own
opinion.

**This is the point at which a benchmark starts lying to you**, so it is worth
being blunt about the cost: that exclusion drops `insert+delete` from 38 scored
cases to 8. The remaining 8 pass, and the class now reads `structure 100% (8/38)`
rather than `100%`. Two tests guard the rule from being widened until nothing is
left — one asserts the metric still scores whole classes, the other that it still
*fails* the class it should.

**What is still not solved.** `adjacent-edit` scores 0% on `structure`, over all
40 of its cases, and that one is real: it injects a tool that appears nowhere in
the other run, so "nothing here corresponds" is the only true description rather
than one of two. The diff pairs those steps anyway because two gap blocks cost
−2.9 and two weak substitutions cost −1.2 — and it cannot simply be made to
prefer gapping, because "one step replaced by an unrelated step" is locally
identical to a tool substitution, which the `substitute` class exists to check it
pairs. The two expectations contradict each other.

Which *side* of a swap is "the one that moved" also stays ambiguous — "steps 1–2
happened later" and "steps 3–4 happened earlier" describe one event, the two
backbones are the same length, and the tie-break is arbitrary. The tests assert
the pairing, which is not ambiguous, and explicitly allow either labelling.

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
        └────► diff (Needleman–Wunsch + moves) ◄────────┘
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
reproducible. No number is reported without a baseline to compare it against, and
none is reported without its limits.

```bash
flightrec bench                 # ~1 minute, 40 runs per metric
flightrec bench --json          # machine-readable
```

| Metric | Measured | Baseline | Target | |
|---|---|---|---|---|
| **Replay fidelity** | **100%** (40/40) | 20% — re-running the task without the recording | 100% vs ~0% | met |
| **Divergence localization** | **100%** over 10 mutation classes (398/398) | 61% — index-by-index `zip()` pairing | alignment ≫ zip | met, and one metric still fails |
| **Replay cost saving** | **25%** cutting at the midpoint, **77%** cutting at 90% | 0% — re-running from step 0 | — | — |
| **Overhead** | **~+95%** wall clock, **~8µs** per span | uninstrumented agent | < 5% wall clock | **missed** |

The first three are exact and reproduce on any machine — they are counts, and the
seeds are fixed. The overhead row is a timing measurement and moves by a few
percent between runs and considerably more between machines; it is quoted with a
`~` for that reason rather than to round it in a flattering direction.

Five things these numbers do not say, in descending order of how much they matter:

**The 61% baseline is not a fair fight.** Half the mutation classes change no
lengths at all, and on those index pairing is *correct* — it scores 100%, which
drags the baseline up from the 0% it scores on insertions and deletions. Read the
per-class lines in `flightrec bench`, not the average; the single number exists
because the table asked for one.

**One number is the wrong thing to read here, and the harness prints four.**
Localization is only "was the changed step paired with its counterpart". Three
others disagree with it, deliberately:

| Metric | Question | Worst class |
|---|---|---|
| `localized` | was the changed step paired with its counterpart? | 100% everywhere |
| `pairing` | was *every* surviving step paired correctly? | 100% everywhere |
| `blame` | is the first divergence reported the real one? | 100% everywhere |
| `structure` | are added and removed steps reported as added and removed? | **adjacent-edit, 0%** |

`blame` and `structure` do not apply to every mutant and print their
denominators for that reason — `structure 100% (8/38)` is a different claim from
`structure 100%`, and printing the first as the second is how a number gets
quoted without its limits.

A blank column means the metric has no answer for that class — not a perfect
score. `structure` is the one that still fails, and
[Where the alignment gets it wrong](#where-the-alignment-gets-it-wrong) explains
why it is a genuine trade-off rather than a bug to fix.

**These numbers have been revised down twice by making the corpus harsher**, and
each revision found something. The generator started with one mutation class and
scored 100%; six classes took it to 95% and broke the aligner in two ways; ten
classes and a fourth metric took it to 99.5% and exposed a case the other three
metrics were structurally unable to see. The three step-level metrics are back at
100% after fixing what that found — `structure` is not, and it is not going to be.

**The overhead target was missed by a factor of about twenty, and the percentage
is the wrong number to read.** Every step of the demo agent is local and finishes
in microseconds, so a fixed per-span cost lands on a run that does almost nothing
— roughly 0.2ms instrumented against 0.1ms bare. The portable figure is ~8µs per
span, which stays under the 5% target for any run longer than about 2ms. One real
model call is three orders of magnitude past that. The percentage is reported
anyway because it is what the target asked for, and moving the goalposts after
seeing the result is how benchmarks become press releases.

**The replay saving is smaller than "skip half the steps" suggests, for a
structural reason.** A model call's prompt carries the whole transcript so far,
so the second half of a run costs far more than the first. Cutting at the
midpoint skips half the steps and a quarter of the tokens. The saving scales with
*where* you cut, not how many steps you skip — hence both numbers in the table.

**Replay fidelity is trajectory-identical, not byte-identical.** Span IDs and
timestamps are excluded from the comparison because a replay cannot recover the
recording's UUIDs or its wall clock. Replays of one recording *are* byte-identical
to each other; that is tested separately.

**The fidelity baseline is generous to the no-tooling case.** It re-runs the task
with fresh RNG streams — "run it again and watch". It scores 20% rather than 0%
because a quarter of runs happen to fire no faults and land on the same clean
trajectory. Against a real provider, which offers no seed at all, the baseline
would be worse than this rather than better.

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
- [x] 8. Run diff with sequence alignment
- [x] 9. Measurement harness → real numbers in this README

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

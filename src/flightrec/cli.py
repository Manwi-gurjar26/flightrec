"""Command-line entry point for flightrec."""

from __future__ import annotations

import argparse

from flightrec import __version__

DEFAULT_DB = "flightrec.db"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flightrec",
        description="A flight recorder for LLM agents: trace, replay, and diff agent runs.",
    )
    parser.add_argument("--version", action="version", version=f"flightrec {__version__}")
    parser.set_defaults(func=None)

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    demo = sub.add_parser("demo", help="run the example agent and record it")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--temperature", type=float, default=0.0)
    demo.add_argument("--faults", choices=["none", "realistic"], default="realistic")
    demo.add_argument(
        "--cities",
        type=int,
        default=2,
        metavar="N",
        help="how many cities the task asks about; each one costs the agent about "
             "four steps, so this is the dial for run length",
    )
    demo.add_argument("--out", help="write the recording to a JSONL file")
    demo.add_argument("--db", help="also store the run in this database")
    demo.set_defaults(func=_cmd_demo)

    serve = sub.add_parser("serve", help="start the collector")
    serve.add_argument("--db", default=DEFAULT_DB)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=_cmd_serve)

    runs = sub.add_parser("runs", help="list recorded runs")
    runs.add_argument("--db", default=DEFAULT_DB)
    runs.add_argument("--limit", type=int, default=20)
    runs.set_defaults(func=_cmd_runs)

    show = sub.add_parser("show", help="print one recorded run as a timeline")
    show.add_argument("run_id")
    show.add_argument("--db", default=DEFAULT_DB)
    show.set_defaults(func=_cmd_show)

    replay = sub.add_parser("replay", help="replay a recorded run, optionally from step N")
    replay.add_argument("run_id")
    replay.add_argument("--db", default=DEFAULT_DB)
    replay.add_argument(
        "--from-step",
        type=int,
        dest="from_step",
        metavar="N",
        help="re-execute live from this step onward; earlier steps come from the recording",
    )
    replay.add_argument(
        "--strict",
        action="store_true",
        help="stop at the edit point instead of executing forward",
    )
    replay.add_argument("--task", help="replay with a different task prompt")
    replay.add_argument("--temperature", type=float, help="replay at a different temperature")
    replay.add_argument("--store", action="store_true", help="save the replay to the database")
    replay.set_defaults(func=_cmd_replay)

    diff = sub.add_parser("diff", help="align two recorded runs and find where they diverged")
    diff.add_argument("run_id")
    diff.add_argument("against", help="the run to compare against")
    diff.add_argument("--db", default=DEFAULT_DB)
    diff.add_argument(
        "--by-index",
        action="store_true",
        dest="by_index",
        help="pair step i with step i instead of aligning -- the baseline, for comparison",
    )
    diff.set_defaults(func=_cmd_diff)

    flaky = sub.add_parser(
        "flaky", help="run a task many times and rank its steps by how much they vary"
    )
    flaky.add_argument("--runs", type=int, default=50, help="how many times to run it")
    flaky.add_argument("--cities", type=int, default=2, help="task length")
    flaky.add_argument(
        "--faults", choices=["none", "realistic"], default="realistic"
    )
    flaky.add_argument("--db", help="report on stored runs instead of fresh ones")
    flaky.set_defaults(func=_cmd_flaky)

    bench = sub.add_parser("bench", help="reproduce every number in the README")
    bench.add_argument("--runs", type=int, default=40, help="recordings per metric")
    bench.add_argument("--repeats", type=int, default=40, help="timing samples for overhead")
    bench.add_argument("--json", action="store_true", help="emit machine-readable results")
    bench.set_defaults(func=_cmd_bench)

    cost = sub.add_parser("cost", help="break down a run's spend, or compare two")
    cost.add_argument("run_id")
    cost.add_argument("against", nargs="?", help="baseline run to compare against")
    cost.add_argument("--db", default=DEFAULT_DB)
    cost.set_defaults(func=_cmd_cost)

    return parser


def _cmd_demo(args: argparse.Namespace) -> int:
    from flightrec.demo.agent import ResearchAgent
    from flightrec.demo.report import format_run
    from flightrec.demo.tools import FaultConfig
    from flightrec.sinks import JSONLSink, MemorySink, TeeSink

    faults = FaultConfig.realistic() if args.faults == "realistic" else FaultConfig()
    memory = MemorySink()
    sink = TeeSink(memory, JSONLSink(args.out)) if args.out else memory

    from flightrec.demo.tools import CITY_DAYS

    available = list(CITY_DAYS)
    if not 1 <= args.cities <= len(available):
        print(f"--cities must be between 1 and {len(available)}")
        return 1

    agent = ResearchAgent(
        seed=args.seed,
        temperature=args.temperature,
        faults=faults,
        sink=sink,
        cities=available[: args.cities],
    )
    result = agent.run()
    result.run.spans = list(memory.spans)

    print(format_run(result, args.seed, args.faults, args.temperature))

    if args.out:
        sink.close()
        print(f"recorded: {args.out}")
    if args.db:
        from flightrec.storage import RunStore

        store = RunStore(args.db)
        store.add_run(result.run)
        store.close()
        print(f"stored:   {args.db}  run_id={result.run.run_id}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from flightrec.collector import create_app

    print(f"collector on http://{args.host}:{args.port}  (db: {args.db})")
    uvicorn.run(create_app(db_path=args.db), host=args.host, port=args.port)
    return 0


def _cmd_runs(args: argparse.Namespace) -> int:
    from flightrec.storage import RunStore

    store = RunStore(args.db)
    summaries = store.list_runs(limit=args.limit)
    if not summaries:
        print(f"no runs in {args.db}")
        return 0

    print(f"{'RUN ID':<34} {'STATUS':<8} {'STEPS':>5} {'ERR':>4} {'TOKENS':>7}  ROOT")
    for s in summaries:
        print(
            f"{s.run_id:<34} {s.status:<8} {s.step_count:>5} {s.error_count:>4} "
            f"{s.total_tokens:>7}  {s.root_name or '-'}"
        )
    store.close()
    return 0


def _cmd_cost(args: argparse.Namespace) -> int:
    from flightrec.pricing import format_usd
    from flightrec.rollup import CostComparison, build_rollup
    from flightrec.storage import RunStore

    store = RunStore(args.db)
    run = store.get_run(args.run_id)
    baseline_run = store.get_run(args.against) if args.against else None
    store.close()

    if run is None:
        print(f"no run {args.run_id!r} in {args.db}")
        return 1
    if args.against and baseline_run is None:
        print(f"no run {args.against!r} in {args.db}")
        return 1

    rollup = build_rollup(run)

    def table(title: str, lines: list) -> None:
        if not lines:
            return
        print(f"\n{title}")
        print(f"  {'':<20} {'CALLS':>6} {'TOKENS':>8} {'TIME':>9} {'COST':>12} {'SHARE':>7}")
        for line in lines:
            from flightrec.web import format_duration

            print(
                f"  {line.label:<20} {line.calls:>6} {line.tokens:>8} "
                f"{format_duration(line.duration_ms):>9} "
                f"{format_usd(line.cost_usd):>12} "
                f"{line.share_of(rollup.total_cost_usd):>6}%"
            )

    print(f"run {rollup.run_id}")
    print(
        f"  {rollup.total_tokens:,} tokens "
        f"({rollup.total_input_tokens:,} in / {rollup.total_output_tokens:,} out)   "
        f"{format_usd(rollup.total_cost_usd)}"
    )
    if not rollup.priced_completely:
        print(
            f"  WARNING: {rollup.unpriced_calls} call(s) used an unpriced model, "
            f"covering {rollup.unpriced_tokens:,} tokens -- this total is incomplete"
        )

    table("BY KIND", rollup.by_kind)
    table("BY MODEL", rollup.by_model)
    table("BY TOOL", rollup.by_tool)

    print("\nATTRIBUTED TO GOING WRONG")
    print(f"  failed steps        {rollup.error_count}")
    print(f"  retries             {rollup.retry_count}")
    print(
        f"  after first failure {rollup.post_failure_calls} call(s), "
        f"{rollup.post_failure_tokens:,} tokens, "
        f"{format_usd(rollup.post_failure_cost_usd)} "
        f"({rollup.post_failure_share}% of the run) -- upper bound"
    )

    if baseline_run is not None:
        comparison = CostComparison(build_rollup(baseline_run), rollup)
        ratio = comparison.cost_ratio
        print(f"\nVERSUS {baseline_run.run_id}")
        print(
            f"  tokens {comparison.token_delta:+,}   "
            f"cost {format_usd(abs(comparison.cost_delta))} "
            f"{'more' if comparison.cost_delta >= 0 else 'less'}"
            + (f"   ({ratio}x)" if ratio else "")
        )
        movers = [row for row in comparison.by_model_delta() + comparison.by_tool_delta() if row[1] or row[2]]
        if movers:
            print("  biggest movers:")
            for label, cost_delta, token_delta in movers[:6]:
                print(
                    f"    {label:<20} {format_usd(abs(cost_delta)):>12} "
                    f"{'+' if cost_delta >= 0 else '-'}   {token_delta:+,} tokens"
                )
        else:
            print("  no per-model or per-tool difference")
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    from flightrec.replay import ReplayMismatch, replay_run
    from flightrec.spans import FR_DIVERGENT, FR_SERVED, SpanStatus
    from flightrec.storage import RunStore

    store = RunStore(args.db)
    original = store.get_run(args.run_id)
    if original is None:
        store.close()
        print(f"no run {args.run_id!r} in {args.db}")
        return 1

    try:
        replay = replay_run(
            original,
            from_step=args.from_step,
            strict=args.strict,
            task=args.task,
            temperature=args.temperature,
        )
    except ReplayMismatch as exc:
        store.close()
        print(f"REPLAY FAILED: {exc}")
        return 2

    print(f"replaying {original.run_id} -> {replay.run.run_id}")
    for index, step in enumerate(replay.run.steps()):
        # Provenance per step, and conflating these would be the exact confusion
        # this tool exists to prevent. "recorded" was read back out of the
        # recording and cost nothing; "live" was really executed because the
        # recording no longer applies; "stopped" is where --strict gave up
        # rather than guess, and is not a step that ran at all.
        if step.attr(FR_SERVED):
            source = "recorded"
        elif "ReplayStopped" in (step.status_message or ""):
            source = "stopped"
        else:
            source = "live"
        marker = "!" if step.status is SpanStatus.ERROR else " "
        print(f" {marker} [{index:>2}] {source:<8} {step.name:<18} {_outcome_text(step)[:56]}")

    print(
        f"\n{replay.served} step(s) served from the recording, "
        f"{replay.live} executed live"
    )
    if replay.stopped:
        print("stopped at the edit point (--strict); nothing past it was executed")
    elif replay.edits or replay.from_step is not None:
        divergence = replay.divergence_step
        print(
            f"first divergence from the recording: step {divergence}"
            if divergence is not None
            else "the edit changed nothing: the trajectory is identical"
        )
    elif replay.faithful:
        print("FAITHFUL: the replayed trajectory is identical to the recording")
    else:
        print(f"NOT FAITHFUL: diverged at step {replay.divergence_step}")

    if args.store:
        store.add_run(replay.run)
        print(f"stored:   {args.db}  run_id={replay.run.run_id}")
    store.close()
    return 0


#: One character per alignment op. The two that are *not* divergences read as
#: quiet punctuation, so a run of them does not compete for attention with the
#: ones that are.
_DIFF_MARKS = {
    "match": "=",
    "cosmetic": "~",
    "changed": "!",
    "replaced": "x",
    "removed": "-",
    "inserted": "+",
}


def _cmd_diff(args: argparse.Namespace) -> int:
    from flightrec.diff import Op, diff_runs, diff_runs_by_index
    from flightrec.storage import RunStore

    store = RunStore(args.db)
    left = store.get_run(args.run_id)
    right = store.get_run(args.against)
    store.close()

    for run_id, run in ((args.run_id, left), (args.against, right)):
        if run is None:
            print(f"no run {run_id!r} in {args.db}")
            return 1

    diff = (diff_runs_by_index if args.by_index else diff_runs)(left, right)

    print(f"diff {left.run_id} -> {right.run_id}")
    if args.by_index:
        print("pairing: index-by-index (baseline; use without --by-index for alignment)")
    print(f"{'':6} {'':2} {'LEFT':<22} {'RIGHT':<22}")

    for column in diff.columns:
        # A move is orthogonal to the content class, so it gets its own marker
        # rather than a sixth op: a step can be reordered *and* changed, and
        # showing only one of those would hide half of what happened.
        mark = ">" if column.moved else _DIFF_MARKS[column.op.value]
        index = column.index
        left_name = column.left.name if column.left else ""
        right_name = column.right.name if column.right else ""
        detail = ""
        if column.op is Op.CHANGED:
            detail = (
                f"{_short(_outcome_text(column.left))}"
                f"  ->  {_short(_outcome_text(column.right))}"
            )
        elif column.op is Op.REPLACED:
            detail = "a different step, not a different result"
        elif column.op is Op.COSMETIC:
            detail = "reworded, same result"
        if column.moved:
            detail = f"moved to step {column.right_index}" + (
                f"; {detail}" if detail else ""
            )
        print(f" [{index:>2}]  {mark}  {left_name:<22} {right_name:<22} {detail}")

    counts = diff.counts
    print(
        f"\n{counts[Op.MATCH]} identical, {counts[Op.COSMETIC]} cosmetic, "
        f"{counts[Op.CHANGED]} changed, {counts[Op.REPLACED]} replaced, "
        f"{counts[Op.REMOVED]} only in left, {counts[Op.INSERTED]} only in right, "
        f"{diff.moved_count} reordered"
    )

    divergence = diff.first_divergence
    if divergence is None:
        print(
            "no genuine divergence: the runs differ only in wording"
            if counts[Op.COSMETIC]
            else "the two runs are identical"
        )
    else:
        step = divergence.left or divergence.right
        # A moved step's op is usually MATCH -- its content is identical, it just
        # happened elsewhere. Printing "match" as the reason for a divergence
        # reads as a contradiction, so the move is named instead.
        if divergence.moved:
            kind = "reordered and changed" if divergence.op is Op.CHANGED else "reordered"
        else:
            kind = divergence.op.value
        print(f"first genuine divergence: step {divergence.index} ({kind}, {step.name})")
    return 0


def _outcome_text(span: "Span") -> str:
    """What a step produced, for one line of terminal output.

    Two traps here, and the obvious one-liner falls into both.

    ``attr(FR_OUTPUT) or span.status_message`` treats an empty list as absent --
    and in this demo an empty list is the *entire failure*, the search that
    returned nothing and sent the agent off inventing data. Rendering the root
    cause as blank space is the kind of quiet omission the timeline exists to
    prevent.

    Testing presence instead is not enough either: a failed tool call records an
    output of ``None`` deliberately, so presence alone prints "None" where the
    error message belongs. A step that errored is described by its error.
    """
    from flightrec.spans import FR_OUTPUT, SpanStatus

    if span.status is SpanStatus.ERROR and span.status_message:
        return span.status_message
    if FR_OUTPUT in span.attributes:
        return str(span.attributes[FR_OUTPUT])
    return span.status_message or ""


def _short(value: object, width: int = 24) -> str:
    # ASCII ellipsis: the Windows console is frequently cp1252 and turns a real
    # one into a replacement character.
    text = str(value if value is not None else "")
    return text if len(text) <= width else text[: width - 3] + "..."


def _cmd_flaky(args: argparse.Namespace) -> int:
    from flightrec.demo.agent import ResearchAgent
    from flightrec.demo.tools import CITY_DAYS, FaultConfig
    from flightrec.flaky import flaky_report, reference_run

    if args.db:
        from flightrec.storage import RunStore

        store = RunStore(args.db)
        runs = [
            run
            for summary in store.list_runs(limit=args.runs)
            if (run := store.get_run(summary.run_id)) is not None
        ]
        store.close()
        source = f"{len(runs)} stored run(s) from {args.db}"
    else:
        faults = FaultConfig.realistic() if args.faults == "realistic" else FaultConfig()
        cities = list(CITY_DAYS)[: args.cities]
        runs = [
            ResearchAgent(seed=seed, faults=faults, cities=cities).run().run
            for seed in range(args.runs)
        ]
        source = f"{args.runs} fresh runs, {len(cities)} cities, faults={args.faults}"

    if len(runs) < 2:
        print("need at least two runs to compare")
        return 1

    report = flaky_report(runs)
    reference = reference_run(runs)

    print(f"flaky steps over {source}")
    print(f"aligned against a {len(reference.steps())}-step reference")
    print()
    for entry in report:
        print(f"  {entry.line()}")

    worst = report[0]
    print()
    print(
        f"most variable: step {worst.position} ({worst.name}), "
        f"{worst.disagreement * 100:.0f}% of runs did something other than its usual thing"
    )
    print(
        "a step at 0% is settled; a high one is a coin toss the agent is "
        "winning most of the time"
    )
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from flightrec.bench import run_bench

    measurements = run_bench(runs=args.runs, repeats=args.repeats)

    if args.json:
        import json

        print(json.dumps([m.as_dict() for m in measurements], indent=2))
        return 0

    print(f"flightrec bench   {args.runs} runs per metric\n")
    for measurement in measurements:
        print(f"{measurement.name}")
        print(
            f"  {measurement.format_value():>8}   vs {measurement.format_baseline():>8} "
            f"({measurement.baseline_label})"
        )
        print(f"  {measurement.detail}")
        for line in measurement.breakdown:
            print(f"    {line}")
        # The caveat is not a footnote. A number whose limits are printed
        # somewhere else is a number that will be quoted without them.
        for line in _wrap(measurement.caveat, 74):
            print(f"    {line}")
        print()
    return 0


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width) if text else []


def _cmd_show(args: argparse.Namespace) -> int:
    from flightrec.spans import SpanStatus
    from flightrec.storage import RunStore

    store = RunStore(args.db)
    run = store.get_run(args.run_id)
    store.close()
    if run is None:
        print(f"no run {args.run_id!r} in {args.db}")
        return 1

    print(f"run {run.run_id}   {len(run.spans)} spans   {run.total_tokens} tokens")
    for step in run.steps():
        marker = "!" if step.status is SpanStatus.ERROR else " "
        print(f" {marker} [{step.sequence:>2}] {step.name:<18} {_outcome_text(step)[:70]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None or args.func is None:
        parser.print_help()
        return 0

    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

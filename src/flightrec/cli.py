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

    # Subcommands are registered as each build step lands:
    #   diff    - align and diff two recorded runs      (step 8)
    #   bench   - reproduce the numbers in the README   (step 9)
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    demo = sub.add_parser("demo", help="run the example agent and record it")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--temperature", type=float, default=0.0)
    demo.add_argument("--faults", choices=["none", "realistic"], default="realistic")
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

    agent = ResearchAgent(
        seed=args.seed,
        temperature=args.temperature,
        faults=faults,
        sink=sink,
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
    from flightrec.spans import FR_DIVERGENT, FR_OUTPUT, FR_SERVED, SpanStatus
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
        # Three distinct provenances, and conflating them would be the exact
        # confusion this tool exists to prevent: "recorded" means the result was
        # read back out of the recording, "re-run" means it was re-executed and
        # is expected to match, "live" means the recording no longer applies.
        if step.attr(FR_DIVERGENT):
            source = "live"
        elif step.attr(FR_SERVED):
            source = "recorded"
        else:
            source = "re-run"
        marker = "!" if step.status is SpanStatus.ERROR else " "
        output = str(step.attr(FR_OUTPUT) or step.status_message or "")
        print(f" {marker} [{index:>2}] {source:<8} {step.name:<18} {output[:56]}")

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


def _cmd_show(args: argparse.Namespace) -> int:
    from flightrec.spans import FR_OUTPUT, SpanStatus
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
        output = str(step.attr(FR_OUTPUT) or step.status_message or "")
        print(f" {marker} [{step.sequence:>2}] {step.name:<18} {output[:70]}")
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

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
    #   replay  - replay a recorded run from step N     (step 7)
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

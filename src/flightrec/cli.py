"""Command-line entry point for flightrec."""

from __future__ import annotations

import argparse

from flightrec import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flightrec",
        description="A flight recorder for LLM agents: trace, replay, and diff agent runs.",
    )
    parser.add_argument("--version", action="version", version=f"flightrec {__version__}")
    parser.set_defaults(func=None)

    # Subcommands are registered as each build step lands:
    #   serve   - start the collector + timeline UI     (steps 4-5)
    #   runs    - list recorded runs                    (step 4)
    #   replay  - replay a recorded run from step N     (step 7)
    #   diff    - align and diff two recorded runs      (step 8)
    #   bench   - reproduce the numbers in the README   (step 9)
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    demo = sub.add_parser("demo", help="run the example agent and record it")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--temperature", type=float, default=0.0)
    demo.add_argument("--faults", choices=["none", "realistic"], default="realistic")
    demo.add_argument("--out", help="write the recording to a JSONL file")
    demo.set_defaults(func=_cmd_demo)

    return parser


def _cmd_demo(args: argparse.Namespace) -> int:
    from flightrec.demo.agent import ResearchAgent
    from flightrec.demo.report import format_run
    from flightrec.demo.tools import FaultConfig
    from flightrec.sinks import JSONLSink, MemorySink

    faults = FaultConfig.realistic() if args.faults == "realistic" else FaultConfig()
    sink = JSONLSink(args.out) if args.out else MemorySink()

    result = ResearchAgent(
        seed=args.seed,
        temperature=args.temperature,
        faults=faults,
        sink=sink,
    ).run()

    print(format_run(result, args.seed, args.faults, args.temperature))
    if args.out:
        sink.close()
        print(f"recorded: {args.out}")
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

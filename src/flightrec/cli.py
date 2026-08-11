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
    #   demo    - run the example agent and record it   (step 3)
    #   serve   - start the collector + timeline UI     (steps 4-5)
    #   runs    - list recorded runs                    (step 4)
    #   replay  - replay a recorded run from step N     (step 7)
    #   diff    - align and diff two recorded runs      (step 8)
    #   bench   - reproduce the numbers in the README   (step 9)
    parser.add_subparsers(dest="command", metavar="<command>")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

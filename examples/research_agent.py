"""Run the demo agent and print what it did.

    python examples/research_agent.py --seed 0 --faults realistic

Equivalent to ``flightrec demo``. Kept as a script because the fastest way to
understand a tracing SDK is to read the handful of lines that use it.
"""

from __future__ import annotations

import sys

from flightrec.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["demo", *sys.argv[1:]]))

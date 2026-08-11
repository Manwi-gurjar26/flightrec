"""Seams for every source of non-determinism in a recorded run.

The single most important design rule in this project: **nothing outside this
module may call ``time.time()``, ``uuid4()`` or ``random`` directly.** Every
source of variation goes through an object that can be swapped for a recorded,
reproducible version at replay time.

Retrofitting this later does not work. A replay is only trustworthy if the
*whole* system was built to be pinned, and the failure mode of a partial
implementation is silent: the replay looks fine and simply is not one.
"""

from __future__ import annotations

import random
import time
import uuid
from typing import Protocol


class Clock(Protocol):
    """Wall-clock time, in epoch seconds."""

    def now(self) -> float: ...


class SystemClock:
    """Real time, at high resolution. Used when recording.

    Not simply ``time.time()``. On Windows that has a resolution of about
    15.6ms, and agent steps routinely complete in microseconds -- so whole
    groups of spans get *identical* start times, every sub-millisecond duration
    reads as exactly zero, and any ordering that falls back on a tiebreaker is
    scrambled. This cost us a real bug: the first recorded trajectory came out
    with the calculator call ahead of the page fetch that produced its operands.

    The fix is the standard one: anchor to the wall clock once, then advance
    using a monotonic high-resolution counter. The result still reads as epoch
    seconds, but it is precise and cannot go backwards when the system clock is
    adjusted mid-run.
    """

    def __init__(self) -> None:
        self._epoch_anchor = time.time()
        self._perf_anchor = time.perf_counter()

    def now(self) -> float:
        return self._epoch_anchor + (time.perf_counter() - self._perf_anchor)


class VirtualClock:
    """Deterministic time. Used when replaying.

    Advances by a fixed step on every read, so a replayed run produces the same
    timestamps in the same order regardless of how fast the machine is. Seeded
    from the recording's own start time so replayed timestamps stay plausible.
    """

    def __init__(self, start: float = 0.0, step: float = 0.001) -> None:
        self._t = start
        self._step = step

    def now(self) -> float:
        value = self._t
        self._t += self._step
        return value


class IdGenerator(Protocol):
    """Span and trace identifiers."""

    def new_id(self) -> str: ...


class RandomIdGenerator:
    """Real UUIDs. Used when recording."""

    def new_id(self) -> str:
        return uuid.uuid4().hex


class SeededIdGenerator:
    """Reproducible identifiers. Used when replaying.

    Span IDs leak into diffs and logs, so an unpinned ID generator makes two
    otherwise-identical runs compare as different. A seeded PRNG over the same
    128-bit space keeps IDs stable across replays without changing their shape.
    """

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def new_id(self) -> str:
        return "%032x" % self._rng.getrandbits(128)

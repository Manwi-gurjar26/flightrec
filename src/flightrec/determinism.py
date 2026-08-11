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
    """Real time. Used when recording."""

    def now(self) -> float:
        return time.time()


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

"""Gate 8: consequence-dependent temporal resolution.

Two events have already been admitted. A strict temporal-resource budget allows
only 16 bins total across the pair.

Each event contains a fine temporal relation: channel A occurs just before B or
B just before A. At 12 bins this order is recoverable. At 8 or 4 bins the same
two pulses collapse into one temporal bin and order is lost.

The successful policy allocates 12 bins to the more consequential event and 4
to the less consequential event. Consequence magnitude is independent of the
event's answer, and the consequential slot swaps across trials.

This gate tests adaptive timing allocation, not biological implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EVENT_LENGTH = 24
FINE_RESOLUTION = 12
COARSE_RESOLUTION = 4
UNIFORM_RESOLUTION = 8
TOTAL_BUDGET = 16

# Every pair is invisible to 8-bin timing but separable at 12 bins.
MICRO_PAIRS = (
    (3, 4),
    (9, 10),
    (15, 16),
    (21, 22),
)


@dataclass(frozen=True)
class TimedEvent:
    order: int
    micro_pair: tuple[int, int]
    consequence: float
    amplitude: float
    relevant: bool


def temporal_bin(position: int, resolution: int) -> int:
    return int(position * resolution // EVENT_LENGTH)


def decode_order(
    event: TimedEvent,
    resolution: int,
) -> int:
    """Return +1/-1 if order survives temporal pooling, else 0."""
    early, late = event.micro_pair

    if event.order == 1:
        a_position, b_position = early, late
    else:
        a_position, b_position = late, early

    a_bin = temporal_bin(a_position, resolution)
    b_bin = temporal_bin(b_position, resolution)

    if a_bin == b_bin:
        return 0

    return 1 if a_bin < b_bin else -1


class AllocationPolicy:
    name = "base"

    def allocate(
        self,
        events: list[TimedEvent],
    ) -> list[int]:
        raise NotImplementedError


class ConsequenceAdaptive(AllocationPolicy):
    name = "consequence_adaptive"

    def allocate(
        self,
        events: list[TimedEvent],
    ) -> list[int]:
        important = int(
            np.argmax([event.consequence for event in events])
        )

        return [
            FINE_RESOLUTION if index == important else COARSE_RESOLUTION
            for index in range(len(events))
        ]


class UniformBudget(AllocationPolicy):
    name = "uniform"

    def allocate(
        self,
        events: list[TimedEvent],
    ) -> list[int]:
        return [UNIFORM_RESOLUTION] * len(events)


class MagnitudeAdaptive(AllocationPolicy):
    name = "magnitude"

    def allocate(
        self,
        events: list[TimedEvent],
    ) -> list[int]:
        largest = int(
            np.argmax([event.amplitude for event in events])
        )

        return [
            FINE_RESOLUTION if index == largest else COARSE_RESOLUTION
            for index in range(len(events))
        ]


class FixedSlotAdaptive(AllocationPolicy):
    name = "fixed_slot"

    def allocate(
        self,
        events: list[TimedEvent],
    ) -> list[int]:
        # Attacker: assumes the first slot is always worth fine timing.
        return [FINE_RESOLUTION, COARSE_RESOLUTION]


class FineEverywhere(AllocationPolicy):
    name = "fine_everywhere"

    def allocate(
        self,
        events: list[TimedEvent],
    ) -> list[int]:
        # Performance control that violates the matched resource budget.
        return [FINE_RESOLUTION] * len(events)


POLICIES = (
    ConsequenceAdaptive,
    UniformBudget,
    MagnitudeAdaptive,
    FixedSlotAdaptive,
    FineEverywhere,
)

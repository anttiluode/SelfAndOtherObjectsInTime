"""Gate 9: online temporal reallocation inside one continuing event.

Gate 8 chose timing resolution once. Gate 9 keeps the same two event identities
alive across four consecutive epochs while consequence swaps between them.

No new event boundary is created at a relevance swap. The successful policy
moves fine timing bandwidth online while preserving the event's resident
context. Epoch durations are also time-warped by x1/x2/x3; decoding is based on
normalized event phase rather than wall-clock samples.

This is a mechanism test, not a literal biological timing model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BASE_LENGTH = 24
FINE_RESOLUTION = 12
COARSE_RESOLUTION = 4
UNIFORM_RESOLUTION = 8
TOTAL_BUDGET = 16
MICRO_PAIRS = (
    (3, 4),
    (9, 10),
    (15, 16),
    (21, 22),
)


@dataclass(frozen=True)
class EpochEvent:
    order: int
    micro_pair: tuple[int, int]
    warp: int
    consequence: float
    amplitude: float
    relevant: bool


def decode_order(
    event: EpochEvent,
    resolution: int,
) -> int:
    early, late = event.micro_pair

    if event.order == 1:
        a_position, b_position = early, late
    else:
        a_position, b_position = late, early

    length = BASE_LENGTH * event.warp
    a_position *= event.warp
    b_position *= event.warp

    a_bin = int(a_position * resolution // length)
    b_bin = int(b_position * resolution // length)

    if a_bin == b_bin:
        return 0

    return 1 if a_bin < b_bin else -1


class OnlinePolicy:
    name = "base"

    def begin_episode(
        self,
        initial_relevant_slot: int,
        context_sign: int,
    ) -> None:
        self.initial_relevant_slot = initial_relevant_slot
        self.context_sign = context_sign
        self.previous_relevant_slot = initial_relevant_slot

    def allocation(
        self,
        events: list[EpochEvent],
        epoch: int,
        current_relevant_slot: int,
    ) -> list[int]:
        raise NotImplementedError

    def context_for_epoch(
        self,
        rng: np.random.Generator,
        epoch: int,
        current_relevant_slot: int,
    ) -> int:
        return self.context_sign

    def finish_epoch(
        self,
        current_relevant_slot: int,
    ) -> None:
        self.previous_relevant_slot = current_relevant_slot


class DynamicConsequencePolicy(OnlinePolicy):
    name = "dynamic_consequence"

    def allocation(
        self,
        events: list[EpochEvent],
        epoch: int,
        current_relevant_slot: int,
    ) -> list[int]:
        important = int(
            np.argmax([event.consequence for event in events])
        )

        return [
            FINE_RESOLUTION if index == important else COARSE_RESOLUTION
            for index in range(2)
        ]


class StaticStartPolicy(OnlinePolicy):
    name = "static_start"

    def allocation(
        self,
        events: list[EpochEvent],
        epoch: int,
        current_relevant_slot: int,
    ) -> list[int]:
        return [
            (
                FINE_RESOLUTION
                if index == self.initial_relevant_slot
                else COARSE_RESOLUTION
            )
            for index in range(2)
        ]


class LagOnePolicy(OnlinePolicy):
    name = "lag_one"

    def allocation(
        self,
        events: list[EpochEvent],
        epoch: int,
        current_relevant_slot: int,
    ) -> list[int]:
        important = (
            current_relevant_slot
            if epoch == 0
            else self.previous_relevant_slot
        )

        return [
            FINE_RESOLUTION if index == important else COARSE_RESOLUTION
            for index in range(2)
        ]


class UniformOnlinePolicy(OnlinePolicy):
    name = "uniform"

    def allocation(
        self,
        events: list[EpochEvent],
        epoch: int,
        current_relevant_slot: int,
    ) -> list[int]:
        return [UNIFORM_RESOLUTION, UNIFORM_RESOLUTION]


class MagnitudeOnlinePolicy(OnlinePolicy):
    name = "magnitude"

    def allocation(
        self,
        events: list[EpochEvent],
        epoch: int,
        current_relevant_slot: int,
    ) -> list[int]:
        largest = int(
            np.argmax([event.amplitude for event in events])
        )

        return [
            FINE_RESOLUTION if index == largest else COARSE_RESOLUTION
            for index in range(2)
        ]


class ResetOnReallocationPolicy(DynamicConsequencePolicy):
    """Gets timing allocation right but treats a relevance swap as a new event."""

    name = "reset_on_reallocation"

    def context_for_epoch(
        self,
        rng: np.random.Generator,
        epoch: int,
        current_relevant_slot: int,
    ) -> int:
        if epoch == 0:
            return self.context_sign

        # If reallocation incorrectly creates a new event, the resident
        # context needed to interpret the temporal relation is no longer
        # available. Model that as an independent replacement context.
        return int(rng.choice(np.array([-1, 1])))


class FineEverywhereOnlinePolicy(OnlinePolicy):
    name = "fine_everywhere"

    def allocation(
        self,
        events: list[EpochEvent],
        epoch: int,
        current_relevant_slot: int,
    ) -> list[int]:
        return [FINE_RESOLUTION, FINE_RESOLUTION]


POLICIES = (
    DynamicConsequencePolicy,
    StaticStartPolicy,
    LagOnePolicy,
    UniformOnlinePolicy,
    MagnitudeOnlinePolicy,
    ResetOnReallocationPolicy,
    FineEverywhereOnlinePolicy,
)

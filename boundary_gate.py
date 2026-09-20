"""Gate 4: discover the event before applying event-relative time.

A long stream contains a quiet resident background plus one relational event at
an unknown onset and duration. The successful mechanism first detects the event
from changes in the stream, then reuses Gate 3's normalized-phase parser.

This is intentionally a hand-designed boundary detector. Gate 4 removes
pre-cut event windows; it does not yet claim that event boundaries are learned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stream_gate import PhaseParser, phase_basis, unit


@dataclass
class ContinuousWorld:
    seed: int = 0
    dim: int = 256
    n_agents: int = 8
    n_objects: int = 10
    noise: float = 0.01

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)

        self.agents = np.stack(
            [unit(rng.normal(size=self.dim)) for _ in range(self.n_agents)]
        )
        self.objects = np.stack(
            [unit(rng.normal(size=self.dim)) for _ in range(self.n_objects)]
        )

        self.background = unit(rng.normal(size=self.dim))
        self.drift = unit(rng.normal(size=self.dim))

        self.roles = {
            name: rng.choice(np.array([-1.0, 1.0]), size=self.dim)
            for name in ("actor", "patient", "view", "object")
        }

        # Evaluator-only labels.
        self.values = rng.choice(
            np.array([-1, 1], dtype=int),
            size=(
                self.n_agents,
                self.n_agents,
                self.n_agents,
                self.n_objects,
            ),
        )

    def event(
        self,
        actor_id: int,
        patient_id: int,
        perspective_id: int,
        object_id: int,
        length: int,
    ) -> np.ndarray:
        basis = phase_basis(length)

        return (
            basis[:, 0, None] * self.agents[perspective_id]
            + 0.8 * basis[:, 1, None] * self.objects[object_id]
            + basis[:, 2, None] * self.agents[actor_id]
            + basis[:, 3, None] * self.agents[patient_id]
        )

    def continuous_stream(
        self,
        actor_id: int,
        patient_id: int,
        perspective_id: int,
        object_id: int,
        total_length: int,
        start: int,
        event_length: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Embed one relational event in quiet resident background."""
        slow_time = np.linspace(-1.0, 1.0, total_length)

        stream = np.stack(
            [
                0.5 * self.background + 0.08 * t * self.drift
                for t in slow_time
            ]
        )

        stream[start : start + event_length] += self.event(
            actor_id,
            patient_id,
            perspective_id,
            object_id,
            event_length,
        )

        stream += self.noise * rng.normal(size=stream.shape)
        return stream


def discover_boundaries(stream: np.ndarray) -> tuple[int, int]:
    """Infer [start, end) from robust change energy.

    The background has only slow drift plus noise. The embedded event introduces
    a boundary step and faster within-event change. We find the first and last
    robustly high derivative. If a very noisy stream yields fewer than two
    threshold crossings, we fall back to the two largest changes.
    """

    change = np.linalg.norm(np.diff(stream, axis=0), axis=1)

    median = float(np.median(change))
    mad = float(np.median(np.abs(change - median))) + 1e-12
    threshold = median + 4.0 * mad

    active = np.flatnonzero(change > threshold)
    if len(active) < 2:
        active = np.argsort(change)[-2:]

    first = int(active.min())
    last = int(active.max())

    # change[first] crosses background -> event.
    # change[last] crosses event -> background.
    return first + 1, last + 1


class ContinuousAssociativeMemory:
    name = "base"

    def __init__(self, world: ContinuousWorld) -> None:
        self.world = world
        self.parser = PhaseParser()
        self.keys: list[np.ndarray] = []
        self.labels: list[float] = []

    def segment(
        self, stream: np.ndarray
    ) -> tuple[np.ndarray, int, int]:
        raise NotImplementedError

    def key(self, stream: np.ndarray) -> np.ndarray:
        event_stream, _, _ = self.segment(stream)
        event = self.parser(event_stream)
        r = self.world.roles

        return unit(
            r["actor"] * event.actor
            + r["patient"] * event.patient
            + r["view"] * event.perspective
            + 0.8 * r["object"] * event.object
        )

    def write(self, stream: np.ndarray, value: int) -> None:
        self.keys.append(self.key(stream))
        self.labels.append(float(value))

    def score(self, stream: np.ndarray) -> float:
        query = self.key(stream)
        keys = np.stack(self.keys)
        similarity = keys @ query

        return float(
            np.dot(np.asarray(self.labels), similarity * similarity)
        )

    def predict(self, stream: np.ndarray) -> int:
        return 1 if self.score(stream) >= 0.0 else -1


class BoundaryPhaseMemory(ContinuousAssociativeMemory):
    """Discover event boundaries, then establish normalized event phase."""

    name = "boundary_phase"

    def segment(
        self, stream: np.ndarray
    ) -> tuple[np.ndarray, int, int]:
        start, end = discover_boundaries(stream)
        return stream[start:end], start, end


class WholeWindowMemory(ContinuousAssociativeMemory):
    """Attacker: pretends the entire ongoing stream is one event."""

    name = "whole_window"

    def segment(
        self, stream: np.ndarray
    ) -> tuple[np.ndarray, int, int]:
        return stream, 0, len(stream)


class FixedWindowMemory(ContinuousAssociativeMemory):
    """Attacker: memorizes the training event location and duration."""

    name = "fixed_window"

    def __init__(
        self,
        world: ContinuousWorld,
        start: int = 16,
        event_length: int = 9,
    ) -> None:
        super().__init__(world)
        self.start = start
        self.event_length = event_length

    def segment(
        self, stream: np.ndarray
    ) -> tuple[np.ndarray, int, int]:
        start = self.start
        end = min(len(stream), start + self.event_length)
        return stream[start:end], start, end


MACHINES = (
    BoundaryPhaseMemory,
    WholeWindowMemory,
    FixedWindowMemory,
)

"""Gate 3: relational roles inferred from temporal modes.

The memory receives one T x D stream. It is not handed actor, patient,
perspective, object, or tau channels. Those components ride different temporal
modes inside the same vector-valued event.

This remains a deliberately synthetic mechanism test, not a literal neural
circuit model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector.copy() if norm == 0.0 else vector / norm


def phase_basis(length: int) -> np.ndarray:
    """Event-normalized temporal basis: view, object, actor, patient."""
    phase = np.linspace(0.0, 1.0, length)

    view = np.ones(length)
    object_mode = np.cos(2.0 * np.pi * phase)
    actor_mode = np.exp(-0.5 * ((phase - 0.25) / 0.10) ** 2)
    patient_mode = np.exp(-0.5 * ((phase - 0.75) / 0.10) ** 2)

    return np.column_stack(
        [view, object_mode, actor_mode, patient_mode]
    )


def fixed_clock_basis(length: int, train_length: int = 9) -> np.ndarray:
    """Attacker basis tied to absolute sample positions from training."""
    phase = np.arange(length, dtype=float) / (train_length - 1)

    view = np.ones(length)
    object_mode = np.cos(2.0 * np.pi * phase)
    actor_mode = np.exp(-0.5 * ((phase - 0.25) / 0.10) ** 2)
    patient_mode = np.exp(-0.5 * ((phase - 0.75) / 0.10) ** 2)

    return np.column_stack(
        [view, object_mode, actor_mode, patient_mode]
    )


class Decomposition(NamedTuple):
    actor: np.ndarray
    patient: np.ndarray
    perspective: np.ndarray
    object: np.ndarray


@dataclass
class StreamWorld:
    seed: int = 0
    dim: int = 256
    n_agents: int = 8
    n_objects: int = 10
    noise: float = 0.015

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)

        self.agents = np.stack(
            [unit(rng.normal(size=self.dim)) for _ in range(self.n_agents)]
        )
        self.objects = np.stack(
            [unit(rng.normal(size=self.dim)) for _ in range(self.n_objects)]
        )

        self.roles = {
            name: rng.choice(np.array([-1.0, 1.0]), size=self.dim)
            for name in ("actor", "patient", "view", "object")
        }

        # Evaluator-only relation labels.
        self.values = rng.choice(
            np.array([-1, 1], dtype=int),
            size=(
                self.n_agents,
                self.n_agents,
                self.n_agents,
                self.n_objects,
            ),
        )

    def event_stream(
        self,
        actor_id: int,
        patient_id: int,
        perspective_id: int,
        object_id: int,
        length: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Superpose all event components into one temporal stream."""
        basis = phase_basis(length)

        stream = (
            basis[:, 0, None] * self.agents[perspective_id]
            + 0.8 * basis[:, 1, None] * self.objects[object_id]
            + basis[:, 2, None] * self.agents[actor_id]
            + basis[:, 3, None] * self.agents[patient_id]
        )

        stream += self.noise * rng.normal(size=stream.shape)
        return stream


class PhaseParser:
    """Infer event roles from normalized event phase."""

    def __call__(self, stream: np.ndarray) -> Decomposition:
        coefficients = np.linalg.pinv(phase_basis(len(stream))) @ stream

        return Decomposition(
            actor=unit(coefficients[2]),
            patient=unit(coefficients[3]),
            perspective=unit(coefficients[0]),
            object=unit(coefficients[1]),
        )


class FixedClockParser:
    """Attacker that assumes training-time absolute sample positions."""

    def __init__(self, train_length: int = 9) -> None:
        self.train_length = train_length

    def __call__(self, stream: np.ndarray) -> Decomposition:
        basis = fixed_clock_basis(
            len(stream), train_length=self.train_length
        )
        coefficients = np.linalg.pinv(basis) @ stream

        return Decomposition(
            actor=unit(coefficients[2]),
            patient=unit(coefficients[3]),
            perspective=unit(coefficients[0]),
            object=unit(coefficients[1]),
        )


class StreamAssociativeMemory:
    name = "base"

    def __init__(
        self,
        world: StreamWorld,
        parser,
        symmetric_pair: bool = False,
    ) -> None:
        self.world = world
        self.parser = parser
        self.symmetric_pair = symmetric_pair
        self.keys: list[np.ndarray] = []
        self.labels: list[float] = []

    def key(self, stream: np.ndarray) -> np.ndarray:
        event = self.parser(stream)
        r = self.world.roles

        if self.symmetric_pair:
            return unit(
                event.actor
                + event.patient
                + r["view"] * event.perspective
                + 0.8 * r["object"] * event.object
            )

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
        similarities = keys @ query

        return float(
            np.dot(np.asarray(self.labels), similarities * similarities)
        )

    def predict(self, stream: np.ndarray) -> int:
        return 1 if self.score(stream) >= 0.0 else -1


class PhaseRelationalMemory(StreamAssociativeMemory):
    name = "phase_relational"

    def __init__(self, world: StreamWorld) -> None:
        super().__init__(world, parser=PhaseParser())


class SymmetricPairMemory(StreamAssociativeMemory):
    name = "symmetric_pair"

    def __init__(self, world: StreamWorld) -> None:
        super().__init__(
            world,
            parser=PhaseParser(),
            symmetric_pair=True,
        )


class FixedClockMemory(StreamAssociativeMemory):
    name = "fixed_clock"

    def __init__(self, world: StreamWorld) -> None:
        super().__init__(
            world,
            parser=FixedClockParser(train_length=9),
        )


MACHINES = (
    PhaseRelationalMemory,
    SymmetricPairMemory,
    FixedClockMemory,
)

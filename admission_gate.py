"""Gate 6: event admission by downstream consequence, not raw magnitude.

A parent process is already active and two candidate interruptions are detected.
Only one child-event slot is available. One interruption is large in local
amplitude but mostly null with respect to the parent's downstream readout. The
other is smaller but projects strongly through a fixed transport into that
readout.

The good mechanism ranks candidates by |m^T J delta|. This gate assumes the
transport J and readout m are known; learning them is explicitly left for a
later gate.

This is a synthetic mechanism test, not a literal brain model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector.copy() if norm == 0.0 else vector / norm


Relation = tuple[int, int, int, int]


@dataclass
class Candidate:
    content: np.ndarray
    delta: np.ndarray
    relevant: bool


@dataclass
class AdmissionWorld:
    seed: int = 0
    dim: int = 128
    n_agents: int = 8
    n_objects: int = 12
    cue_noise: float = 0.03

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
            for name in ("actor", "patient", "view", "object", "child")
        }

        q, _ = np.linalg.qr(rng.normal(size=(self.dim, self.dim)))
        self.transport = q
        self.readout = unit(rng.normal(size=self.dim))
        self.preimage = self.transport.T @ self.readout

    def relation_key(self, relation: Relation) -> np.ndarray:
        actor, patient, perspective, object_id = relation
        r = self.roles

        return unit(
            r["actor"] * self.agents[actor]
            + r["patient"] * self.agents[patient]
            + r["view"] * self.agents[perspective]
            + 0.8 * r["object"] * self.objects[object_id]
        )

    def observe_relation(
        self,
        relation: Relation,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return unit(
            self.relation_key(relation)
            + self.cue_noise * rng.normal(size=self.dim)
        )

    def candidate_delta(
        self,
        relevant: bool,
        rng: np.random.Generator,
        strength: float | None = None,
    ) -> np.ndarray:
        orth = rng.normal(size=self.dim)
        orth -= self.preimage * float(self.preimage @ orth)
        orth = unit(orth)

        if relevant:
            if strength is None:
                strength = float(rng.uniform(0.35, 0.65))
            return strength * self.preimage + 0.05 * orth

        amplitude = float(rng.uniform(2.0, 6.0))
        return amplitude * orth + 0.02 * self.preimage

    def consequence_score(self, candidate: Candidate) -> float:
        return abs(
            float(
                self.readout
                @ (self.transport @ candidate.delta)
            )
        )


class AdmissionMemory:
    name = "base"

    def __init__(self, world: AdmissionWorld) -> None:
        self.world = world
        self.keys: list[np.ndarray] = []
        self.labels: list[float] = []

    def admitted(
        self,
        candidates: list[Candidate],
    ) -> list[tuple[int, float]]:
        raise NotImplementedError

    def compose(
        self,
        parent: np.ndarray,
        candidates: list[Candidate],
    ) -> np.ndarray:
        admitted = self.admitted(candidates)

        if not admitted:
            return unit(parent)

        child = np.zeros_like(parent)

        for index, weight in admitted:
            child += weight * candidates[index].content

        return unit(
            parent
            + 0.8 * self.world.roles["child"] * child
        )

    def write(
        self,
        parent: np.ndarray,
        candidates: list[Candidate],
        value: int,
    ) -> None:
        self.keys.append(self.compose(parent, candidates))
        self.labels.append(float(value))

    def score(
        self,
        parent: np.ndarray,
        candidates: list[Candidate],
    ) -> float:
        query = self.compose(parent, candidates)
        keys = np.stack(self.keys)
        similarity = keys @ query

        return float(
            np.dot(
                np.asarray(self.labels),
                similarity * similarity,
            )
        )

    def predict(
        self,
        parent: np.ndarray,
        candidates: list[Candidate],
    ) -> int:
        return 1 if self.score(parent, candidates) >= 0.0 else -1

    def admission_precision(
        self,
        candidates: list[Candidate],
    ) -> float:
        admitted = self.admitted(candidates)

        if not admitted:
            return 0.0

        relevant = sum(
            1
            for index, _ in admitted
            if candidates[index].relevant
        )

        return float(relevant / len(admitted))


class ConsequenceAdmission(AdmissionMemory):
    name = "consequence"

    def admitted(
        self,
        candidates: list[Candidate],
    ) -> list[tuple[int, float]]:
        scores = [
            self.world.consequence_score(candidate)
            for candidate in candidates
        ]
        index = int(np.argmax(scores))
        return [(index, 1.0)]


class MagnitudeAdmission(AdmissionMemory):
    name = "magnitude"

    def admitted(
        self,
        candidates: list[Candidate],
    ) -> list[tuple[int, float]]:
        magnitudes = [
            float(np.linalg.norm(candidate.delta))
            for candidate in candidates
        ]
        index = int(np.argmax(magnitudes))
        return [(index, 1.0)]


class AdmitAll(AdmissionMemory):
    name = "admit_all"

    def admitted(
        self,
        candidates: list[Candidate],
    ) -> list[tuple[int, float]]:
        return [
            (index, float(np.linalg.norm(candidate.delta)))
            for index, candidate in enumerate(candidates)
        ]


class IgnoreInterruptions(AdmissionMemory):
    name = "ignore"

    def admitted(
        self,
        candidates: list[Candidate],
    ) -> list[tuple[int, float]]:
        return []


MACHINES = (
    ConsequenceAdmission,
    MagnitudeAdmission,
    AdmitAll,
    IgnoreInterruptions,
)

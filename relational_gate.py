"""Gate 2 mechanisms: distributed self/other role binding without agent IDs.

The machine never receives integer agent identifiers. It receives noisy
continuous vectors for actor, patient, current perspective, object, and
event-relative time. Hidden integer IDs exist only inside the synthetic world
and evaluator so we can construct matched tests.

This is a mechanism toy, not a model of consciousness or a claim that the brain
uses these exact binding operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector.copy() if norm == 0.0 else vector / norm


class Observation(NamedTuple):
    actor: np.ndarray
    patient: np.ndarray
    perspective: np.ndarray
    object: np.ndarray
    tau: np.ndarray


@dataclass
class RelationalWorld:
    """Synthetic world with distributed identity, object and time cues."""

    seed: int = 0
    dim: int = 512
    n_agents: int = 8
    n_objects: int = 10
    n_tau: int = 5
    cue_noise: float = 0.02

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.rng = rng

        self.agents = np.stack(
            [unit(rng.normal(size=self.dim)) for _ in range(self.n_agents)]
        )
        self.objects = np.stack(
            [unit(rng.normal(size=self.dim)) for _ in range(self.n_objects)]
        )
        self.times = np.stack(
            [unit(rng.normal(size=self.dim)) for _ in range(self.n_tau)]
        )

        self.roles = {
            name: rng.choice(np.array([-1.0, 1.0]), size=self.dim)
            for name in ("actor", "patient", "view", "object", "time")
        }

        # Evaluator-only hidden labels. These IDs are never passed to memory.
        self.values = rng.choice(
            np.array([-1, 1], dtype=int),
            size=(
                self.n_agents,
                self.n_agents,
                self.n_agents,
                self.n_objects,
                self.n_tau,
            ),
        )

    @property
    def self_anchor(self) -> np.ndarray:
        return self.agents[0].copy()

    def _noisy(
        self, bank: np.ndarray, index: int, rng: np.random.Generator
    ) -> np.ndarray:
        return unit(
            bank[index] + self.cue_noise * rng.normal(size=self.dim)
        )

    def observation(
        self,
        actor_id: int,
        patient_id: int,
        perspective_id: int,
        object_id: int,
        tau_id: int,
        rng: np.random.Generator,
    ) -> Observation:
        """Create a noisy continuous observation from evaluator-side IDs."""
        return Observation(
            actor=self._noisy(self.agents, actor_id, rng),
            patient=self._noisy(self.agents, patient_id, rng),
            perspective=self._noisy(self.agents, perspective_id, rng),
            object=self._noisy(self.objects, object_id, rng),
            tau=self._noisy(self.times, tau_id, rng),
        )


class AssociativeMemory:
    """Content-addressed write/read memory shared by all Gate 2 mechanisms."""

    name = "base"

    def __init__(self, world: RelationalWorld) -> None:
        self.world = world
        self.original_anchor = world.self_anchor
        self.anchor = self.original_anchor.copy()
        self.keys: list[np.ndarray] = []
        self.labels: list[float] = []
        self.residue = np.zeros(world.dim, dtype=float)

    def key(self, observation: Observation) -> np.ndarray:
        raise NotImplementedError

    def write(self, observation: Observation, value: int) -> None:
        key = self.key(observation)
        self.keys.append(key)
        self.labels.append(float(value))
        self.residue *= 0.995
        self.residue += float(value) * key

    def score(self, observation: Observation) -> float:
        query = self.key(observation)
        if not self.keys:
            return 0.0

        keys = np.stack(self.keys)
        similarities = keys @ query

        # Equivalent to q^T (sum y_i k_i k_i^T) q without allocating M.
        return float(
            np.dot(np.asarray(self.labels), similarities * similarities)
        )

    def predict(self, observation: Observation) -> int:
        return 1 if self.score(observation) >= 0.0 else -1

    @property
    def anchor_fidelity(self) -> float:
        return float(unit(self.anchor) @ self.original_anchor)

    @property
    def residue_norm(self) -> float:
        return float(np.linalg.norm(self.residue))


class DistributedRoleBinder(AssociativeMemory):
    """Bind continuous identities into actor/patient/view roles."""

    name = "distributed_binder"

    def __init__(self, world: RelationalWorld) -> None:
        super().__init__(world)
        rng = np.random.default_rng(12_345 + world.dim)
        self.self_role = unit(rng.normal(size=world.dim))

    def key(self, observation: Observation) -> np.ndarray:
        r = self.world.roles
        selfness = max(
            0.0, float(observation.perspective @ self.original_anchor)
        )

        return unit(
            r["actor"] * observation.actor
            + r["patient"] * observation.patient
            + r["view"] * observation.perspective
            + 0.8 * selfness * self.self_role
            + 0.8 * r["object"] * observation.object
            + 0.6 * r["time"] * observation.tau
        )


class RoleBlindBag(AssociativeMemory):
    """Attacker: knows participants but loses actor/patient direction."""

    name = "role_blind_bag"

    def key(self, observation: Observation) -> np.ndarray:
        r = self.world.roles
        return unit(
            observation.actor
            + observation.patient
            + r["view"] * observation.perspective
            + 0.8 * r["object"] * observation.object
            + 0.6 * r["time"] * observation.tau
        )


class ViewlessMemory(AssociativeMemory):
    """Attacker: keeps actor/patient roles but removes perspective."""

    name = "viewless"

    def key(self, observation: Observation) -> np.ndarray:
        r = self.world.roles
        return unit(
            r["actor"] * observation.actor
            + r["patient"] * observation.patient
            + 0.8 * r["object"] * observation.object
            + 0.6 * r["time"] * observation.tau
        )


class SelfOtherBinaryMemory(AssociativeMemory):
    """Attacker: recognizes SELF vs OTHER but collapses all OTHER identities."""

    name = "self_other_binary"

    def __init__(self, world: RelationalWorld) -> None:
        super().__init__(world)
        rng = np.random.default_rng(987_654 + world.dim)
        self.self_view = unit(rng.normal(size=world.dim))
        self.other_view = unit(rng.normal(size=world.dim))

    def key(self, observation: Observation) -> np.ndarray:
        r = self.world.roles
        is_self = (
            float(observation.perspective @ self.original_anchor) > 0.55
        )
        view = self.self_view if is_self else self.other_view

        return unit(
            r["actor"] * observation.actor
            + r["patient"] * observation.patient
            + view
            + 0.8 * r["object"] * observation.object
            + 0.6 * r["time"] * observation.tau
        )


class MutableAnchorBinder(DistributedRoleBinder):
    """Attacker: relational binding works, but visited views rewrite SELF."""

    name = "mutable_anchor"

    def write(self, observation: Observation, value: int) -> None:
        self.anchor = unit(
            0.96 * self.anchor + 0.04 * observation.perspective
        )
        super().write(observation, value)


MACHINES = (
    DistributedRoleBinder,
    RoleBlindBag,
    ViewlessMemory,
    SelfOtherBinaryMemory,
    MutableAnchorBinder,
)

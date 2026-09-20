"""Mechanisms for Gate 1: a perspective detour can be a write.

This is a computational toy, not a model of consciousness or a claim about a
specific brain circuit.  The narrow question is whether a system can preserve a
stable control anchor while retaining a useful residue of a simulated/observed
other perspective.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def unit(vector: np.ndarray) -> np.ndarray:
    """Return a unit-length copy (or the unchanged zero vector)."""
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector.copy() if norm == 0.0 else vector / norm


@dataclass
class TemporalObjectWorld:
    """Fixed distributed addresses for agents, objects, and event-relative time."""

    n_agents: int = 7
    n_objects: int = 11
    dim: int = 64
    n_tau: int = 5
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.agent = np.stack(
            [unit(rng.normal(size=self.dim)) for _ in range(self.n_agents)]
        )
        self.object = np.stack(
            [unit(rng.normal(size=self.dim)) for _ in range(self.n_objects)]
        )
        self.tau = np.stack(
            [unit(rng.normal(size=self.dim)) for _ in range(self.n_tau)]
        )

    def key(self, agent_id: int, object_id: int, tau_bin: int) -> np.ndarray:
        """Bind perspective, object, and event-relative temporal address."""
        return unit(
            self.agent[agent_id]
            + self.object[object_id]
            + 0.7 * self.tau[tau_bin % self.n_tau]
        )


class PerspectiveMachine:
    """Shared resident state and associative write/read machinery."""

    name = "base"

    def __init__(
        self,
        world: TemporalObjectWorld,
        self_id: int = 0,
        decay: float = 0.985,
        learning_rate: float = 1.0,
    ) -> None:
        self.world = world
        self.self_id = self_id
        self.decay = decay
        self.learning_rate = learning_rate

        self.original_anchor = world.agent[self_id].copy()
        self.anchor = self.original_anchor.copy()
        self.active = self.anchor.copy()

        self.memory = np.zeros((world.dim, world.dim), dtype=float)
        self.residue = np.zeros(world.dim, dtype=float)
        self.step = 0

    def _write(
        self, agent_id: int, object_id: int, tau_bin: int, value: int
    ) -> None:
        key = self.world.key(agent_id, object_id, tau_bin)

        self.memory *= self.decay
        self.residue *= self.decay

        self.memory += (
            self.learning_rate * float(value) * np.outer(key, key)
        )
        self.residue += self.learning_rate * float(value) * key
        self.step += 1

    def read_score(self, agent_id: int, object_id: int, tau_bin: int) -> float:
        key = self.world.key(agent_id, object_id, tau_bin)
        return float(key @ self.memory @ key)

    def predict(self, agent_id: int, object_id: int, tau_bin: int) -> int:
        return 1 if self.read_score(agent_id, object_id, tau_bin) >= 0.0 else -1

    @property
    def anchor_fidelity(self) -> float:
        return float(unit(self.anchor) @ self.original_anchor)

    @property
    def residue_norm(self) -> float:
        return float(np.linalg.norm(self.residue))

    def detour(
        self, agent_id: int, object_id: int, tau_bin: int, value: int
    ) -> None:
        raise NotImplementedError


class AnchoredResidueMachine(PerspectiveMachine):
    """Perspective moves; control anchor stays fixed; resident state is written."""

    name = "anchored_residue"

    def detour(
        self, agent_id: int, object_id: int, tau_bin: int, value: int
    ) -> None:
        self.active = self.world.agent[agent_id].copy()
        self._write(agent_id, object_id, tau_bin, value)
        self.active = self.anchor.copy()


class SnapshotResetMachine(PerspectiveMachine):
    """Attacker: perspective is perfectly sandboxed and leaves no residue."""

    name = "snapshot_reset"

    def detour(
        self, agent_id: int, object_id: int, tau_bin: int, value: int
    ) -> None:
        memory_before = self.memory.copy()
        residue_before = self.residue.copy()
        step_before = self.step

        self.active = self.world.agent[agent_id].copy()
        self._write(agent_id, object_id, tau_bin, value)

        self.memory = memory_before
        self.residue = residue_before
        self.step = step_before
        self.active = self.anchor.copy()


class LeakyOverwriteMachine(PerspectiveMachine):
    """Attacker: residue survives, but simulated perspectives rewrite identity."""

    name = "leaky_overwrite"

    def __init__(
        self,
        *args,
        identity_leak: float = 0.08,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.identity_leak = identity_leak

    def detour(
        self, agent_id: int, object_id: int, tau_bin: int, value: int
    ) -> None:
        self.active = self.world.agent[agent_id].copy()
        self.anchor = unit(
            (1.0 - self.identity_leak) * self.anchor
            + self.identity_leak * self.active
        )
        self._write(agent_id, object_id, tau_bin, value)
        self.active = self.anchor.copy()


MACHINES = (
    SnapshotResetMachine,
    AnchoredResidueMachine,
    LeakyOverwriteMachine,
)

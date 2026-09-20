"""Gate 5: nested event stacks.

Gate 4 discovered a single event boundary inside an ongoing stream. Gate 5 asks
what happens when one event interrupts another before the parent event is done.

The boundary markers here are *not* a new oracle for relational content. They
stand in for already-detected open/close transitions. The scientific question
is narrower: does interruption require a stack-like event state so that the
parent's local clock can pause, the child can run on its own local clock, and
the child can write back into the resumed parent?

This is a synthetic mechanism test, not a literal neural circuit model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from stream_gate import PhaseParser, phase_basis, unit


Token = tuple[str, np.ndarray | None]


@dataclass
class NestedWorld:
    seed: int = 0
    dim: int = 256
    n_agents: int = 8
    n_objects: int = 10
    noise: float = 0.015
    child_gain: float = 1.6

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

    def event_stream(
        self,
        actor_id: int,
        patient_id: int,
        perspective_id: int,
        object_id: int,
        length: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        basis = phase_basis(length)

        stream = (
            basis[:, 0, None] * self.agents[perspective_id]
            + 0.8 * basis[:, 1, None] * self.objects[object_id]
            + basis[:, 2, None] * self.agents[actor_id]
            + basis[:, 3, None] * self.agents[patient_id]
        )

        stream += self.noise * rng.normal(size=stream.shape)
        return stream

    def nested_episode(
        self,
        parent: tuple[int, int, int, int],
        child: tuple[int, int, int, int],
        parent_length: int,
        child_length: int,
        child_insert: int,
        rng: np.random.Generator,
    ) -> list[Token]:
        """Create parent -> child -> resumed parent as one token stream."""
        parent_stream = self.event_stream(
            *parent, length=parent_length, rng=rng
        )
        child_stream = self.child_gain * self.event_stream(
            *child, length=child_length, rng=rng
        )

        tokens: list[Token] = [("open_parent", None)]

        tokens.extend(
            ("data", frame)
            for frame in parent_stream[:child_insert]
        )

        tokens.append(("open_child", None))
        tokens.extend(("data", frame) for frame in child_stream)
        tokens.append(("close_child", None))

        tokens.extend(
            ("data", frame)
            for frame in parent_stream[child_insert:]
        )

        tokens.append(("close_parent", None))
        return tokens


def bind_event(
    world: NestedWorld,
    stream: np.ndarray,
) -> np.ndarray:
    """Turn one event-local temporal stream into a relational key."""
    actor, patient, perspective, object_vector = PhaseParser()(stream)
    roles = world.roles

    return unit(
        roles["actor"] * actor
        + roles["patient"] * patient
        + roles["view"] * perspective
        + 0.8 * roles["object"] * object_vector
    )


def split_nested(tokens: list[Token]) -> tuple[np.ndarray, np.ndarray]:
    """Recover parent and child streams while pausing the parent."""
    parent: list[np.ndarray] = []
    child: list[np.ndarray] = []
    in_child = False

    for token, value in tokens:
        if token == "open_child":
            in_child = True
        elif token == "close_child":
            in_child = False
        elif token == "data" and value is not None:
            if in_child:
                child.append(value)
            else:
                parent.append(value)

    if not parent or not child:
        raise ValueError("nested episode must contain parent and child data")

    return np.stack(parent), np.stack(child)


class AssociativeNestedMemory:
    name = "base"

    def __init__(self, world: NestedWorld) -> None:
        self.world = world
        self.keys: list[np.ndarray] = []
        self.labels: list[float] = []

    def key(self, tokens: list[Token]) -> np.ndarray:
        raise NotImplementedError

    def write(self, tokens: list[Token], value: int) -> None:
        self.keys.append(self.key(tokens))
        self.labels.append(float(value))

    def score(self, tokens: list[Token]) -> float:
        query = self.key(tokens)
        keys = np.stack(self.keys)
        similarity = keys @ query

        return float(
            np.dot(np.asarray(self.labels), similarity * similarity)
        )

    def predict(self, tokens: list[Token]) -> int:
        return 1 if self.score(tokens) >= 0.0 else -1


class NestedStackMemory(AssociativeNestedMemory):
    """Pause parent phase, run child phase, then write child into parent."""

    name = "nested_stack"

    def key(self, tokens: list[Token]) -> np.ndarray:
        parent_stream, child_stream = split_nested(tokens)

        parent_key = bind_event(self.world, parent_stream)
        child_key = bind_event(self.world, child_stream)

        # This is the write-back operation. The parent resumes with its own
        # relational state intact, but the completed child now changes the
        # address of the resumed parent.
        return unit(
            parent_key
            + 0.8 * self.world.roles["child"] * child_key
        )


class FlatTimelineMemory(AssociativeNestedMemory):
    """Attacker: erase nesting and force everything onto one clock."""

    name = "flat_timeline"

    def key(self, tokens: list[Token]) -> np.ndarray:
        frames = [
            value
            for token, value in tokens
            if token == "data" and value is not None
        ]
        return bind_event(self.world, np.stack(frames))


class ResetOnResumeMemory(AssociativeNestedMemory):
    """Attacker: child interruption causes parent state to restart."""

    name = "reset_on_resume"

    def key(self, tokens: list[Token]) -> np.ndarray:
        resumed: list[np.ndarray] = []
        after_child = False

        for token, value in tokens:
            if token == "close_child":
                after_child = True
            elif (
                token == "data"
                and value is not None
                and after_child
            ):
                resumed.append(value)

        if len(resumed) < 4:
            raise ValueError("resumed parent segment is too short")

        return bind_event(self.world, np.stack(resumed))


class NoWritebackMemory(AssociativeNestedMemory):
    """Attacker: preserve parent continuity but discard child residue."""

    name = "no_writeback"

    def key(self, tokens: list[Token]) -> np.ndarray:
        parent_stream, _ = split_nested(tokens)
        return bind_event(self.world, parent_stream)


MACHINES = (
    NestedStackMemory,
    FlatTimelineMemory,
    ResetOnResumeMemory,
    NoWritebackMemory,
)

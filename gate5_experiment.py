"""Gate 5: nested event stack.

Training uses one short child interruption at a fixed parent phase. Test episodes
randomize parent duration, child duration, and interruption location.

The matched data are constructed in pairs: the parent event is identical while
the child event changes and the required label flips. Therefore a mechanism
cannot pass by preserving the parent alone; child computation must write back
into the resumed parent.

The boundary markers model the output of a Gate-4-like boundary detector. Gate 5
isolates stack semantics rather than rediscovering boundaries.
"""

from __future__ import annotations

import argparse
import json
from typing import Type

import numpy as np

from nested_gate import (
    MACHINES,
    AssociativeNestedMemory,
    NestedStackMemory,
    NestedWorld,
)


EpisodeSpec = tuple[
    tuple[int, int, int, int],
    tuple[int, int, int, int],
    int,
]


def random_relation(
    world: NestedWorld,
    rng: np.random.Generator,
) -> tuple[int, int, int, int]:
    actor, patient = rng.choice(
        world.n_agents, size=2, replace=False
    )

    return (
        int(actor),
        int(patient),
        int(rng.integers(world.n_agents)),
        int(rng.integers(world.n_objects)),
    )


def make_matched_pairs(
    world: NestedWorld,
    seed: int,
    pairs: int = 20,
) -> list[EpisodeSpec]:
    rng = np.random.default_rng(seed + 10)
    records: list[EpisodeSpec] = []

    for _ in range(pairs):
        parent = random_relation(world, rng)
        child_a = random_relation(world, rng)
        child_b = random_relation(world, rng)

        while child_b == child_a:
            child_b = random_relation(world, rng)

        sign = int(rng.choice(np.array([-1, 1])))

        # Same parent, different child, opposite labels.
        records.extend(
            [
                (parent, child_a, sign),
                (parent, child_b, -sign),
            ]
        )

    return records


def run_one(
    machine_cls: Type[AssociativeNestedMemory],
    seed: int,
) -> dict[str, float]:
    world = NestedWorld(seed=seed)
    machine = machine_cls(world)
    records = make_matched_pairs(world, seed=seed)

    # Training is deliberately regular: one fixed interruption pattern.
    train_rng = np.random.default_rng(seed + 1_000)

    for parent, child, value in records:
        tokens = world.nested_episode(
            parent=parent,
            child=child,
            parent_length=17,
            child_length=7,
            child_insert=8,
            rng=train_rng,
        )
        machine.write(tokens, value)

    test_rng = np.random.default_rng(seed + 2_000)

    correct: list[bool] = []
    long_child_correct: list[bool] = []

    for parent, child, value in records:
        parent_length = int(test_rng.integers(13, 26))
        child_length = int(test_rng.integers(7, 32))

        # Keep at least four parent frames on each side of interruption.
        child_insert = int(
            test_rng.integers(4, parent_length - 4)
        )

        tokens = world.nested_episode(
            parent=parent,
            child=child,
            parent_length=parent_length,
            child_length=child_length,
            child_insert=child_insert,
            rng=test_rng,
        )

        is_correct = machine.predict(tokens) == value
        correct.append(is_correct)

        if child_length >= 20:
            long_child_correct.append(is_correct)

    return {
        "accuracy": float(np.mean(correct)),
        "long_child_accuracy": float(
            np.mean(long_child_correct)
        ),
    }


def summarize(
    machine_cls: Type[AssociativeNestedMemory],
    seeds: int = 64,
) -> dict[str, object]:
    rows = [run_one(machine_cls, seed) for seed in range(seeds)]

    result: dict[str, object] = {}

    for metric in rows[0]:
        values = np.asarray(
            [row[metric] for row in rows],
            dtype=float,
        )

        result[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }

    result["gate_pass"] = bool(
        machine_cls is NestedStackMemory
        and result["accuracy"]["mean"] > 0.95
        and result["long_child_accuracy"]["mean"] > 0.95
    )

    return result


def run_gate(seeds: int = 64) -> dict[str, object]:
    return {
        machine_cls.name: summarize(machine_cls, seeds=seeds)
        for machine_cls in MACHINES
    }


def assert_gate(receipt: dict[str, object]) -> None:
    assert receipt["nested_stack"]["gate_pass"]
    assert not receipt["flat_timeline"]["gate_pass"]
    assert not receipt["reset_on_resume"]["gate_pass"]
    assert not receipt["no_writeback"]["gate_pass"]

    assert receipt["nested_stack"]["accuracy"]["mean"] > 0.95
    assert (
        receipt["nested_stack"]["long_child_accuracy"]["mean"]
        > 0.95
    )

    # Flattening partially works because it still sees all content, but it
    # degrades once child duration moves the parent's later role on a global
    # clock.
    assert receipt["flat_timeline"]["accuracy"]["mean"] < 0.82

    # Resetting loses the parent's pre-interruption role; no-writeback keeps
    # the parent but cannot distinguish paired episodes whose labels differ
    # only by child content.
    assert receipt["reset_on_resume"]["accuracy"]["mean"] < 0.60
    assert receipt["no_writeback"]["accuracy"]["mean"] < 0.60


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--assert-gate", action="store_true")
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    receipt = run_gate(seeds=args.seeds)

    if args.assert_gate:
        assert_gate(receipt)

    text = json.dumps(receipt, indent=2, sort_keys=True)
    print(text)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()

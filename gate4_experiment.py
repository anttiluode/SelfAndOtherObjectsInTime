"""Gate 4: discover the event before timing it.

Training uses an event at a fixed location only so the fixed-window attacker has
the strongest possible chance. Testing randomizes both event onset and duration.

The successful mechanism is never given the event boundaries.
"""

from __future__ import annotations

import argparse
import json
from typing import Type

import numpy as np

from boundary_gate import (
    MACHINES,
    BoundaryPhaseMemory,
    ContinuousAssociativeMemory,
    ContinuousWorld,
)


Record = tuple[int, int, int, int, int]


def make_reversal_pairs(
    world: ContinuousWorld,
    seed: int,
    pairs: int = 24,
) -> list[Record]:
    rng = np.random.default_rng(seed + 11)
    records: list[Record] = []
    seen: set[tuple[int, int, int, int]] = set()

    while len(records) < 2 * pairs:
        actor, patient = rng.choice(
            world.n_agents, size=2, replace=False
        )
        perspective = int(rng.integers(world.n_agents))
        object_id = int(rng.integers(world.n_objects))

        forward = (
            int(actor),
            int(patient),
            perspective,
            object_id,
        )
        reverse = (
            int(patient),
            int(actor),
            perspective,
            object_id,
        )

        if forward in seen or reverse in seen:
            continue

        y_forward = int(world.values[forward])
        y_reverse = int(world.values[reverse])

        if y_forward != y_reverse:
            records.extend(
                [
                    (*forward, y_forward),
                    (*reverse, y_reverse),
                ]
            )
            seen.update((forward, reverse))

    return records


def interval_iou(
    predicted_start: int,
    predicted_end: int,
    true_start: int,
    true_end: int,
) -> float:
    intersection = max(
        0,
        min(predicted_end, true_end)
        - max(predicted_start, true_start),
    )
    union = (
        max(predicted_end, true_end)
        - min(predicted_start, true_start)
    )
    return float(intersection / union)


def run_one(
    machine_cls: Type[ContinuousAssociativeMemory],
    seed: int,
) -> dict[str, float]:
    world = ContinuousWorld(seed=seed)
    machine = machine_cls(world)
    records = make_reversal_pairs(world, seed=seed)

    # Training is deliberately easy for the fixed-window attacker.
    train_rng = np.random.default_rng(seed + 100)
    for actor, patient, perspective, object_id, value in records:
        stream = world.continuous_stream(
            actor,
            patient,
            perspective,
            object_id,
            total_length=64,
            start=16,
            event_length=9,
            rng=train_rng,
        )
        machine.write(stream, value)

    test_rng = np.random.default_rng(seed + 200)
    correct: list[bool] = []
    ious: list[float] = []
    boundary_mae: list[float] = []

    for actor, patient, perspective, object_id, value in records:
        event_length = int(test_rng.integers(9, 26))
        start = int(
            test_rng.integers(5, 64 - event_length - 4)
        )
        end = start + event_length

        stream = world.continuous_stream(
            actor,
            patient,
            perspective,
            object_id,
            total_length=64,
            start=start,
            event_length=event_length,
            rng=test_rng,
        )

        correct.append(machine.predict(stream) == value)

        _, predicted_start, predicted_end = machine.segment(stream)

        ious.append(
            interval_iou(
                predicted_start,
                predicted_end,
                start,
                end,
            )
        )
        boundary_mae.append(
            0.5
            * (
                abs(predicted_start - start)
                + abs(predicted_end - end)
            )
        )

    return {
        "relation_accuracy": float(np.mean(correct)),
        "boundary_iou": float(np.mean(ious)),
        "boundary_mae": float(np.mean(boundary_mae)),
    }


def summarize(
    machine_cls: Type[ContinuousAssociativeMemory],
    seeds: int = 64,
) -> dict[str, object]:
    rows = [run_one(machine_cls, seed) for seed in range(seeds)]

    result: dict[str, object] = {}
    for metric in rows[0]:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        result[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }

    result["gate_pass"] = bool(
        machine_cls is BoundaryPhaseMemory
        and result["relation_accuracy"]["mean"] > 0.82
        and result["boundary_iou"]["mean"] > 0.95
        and result["boundary_mae"]["mean"] < 0.50
    )
    return result


def run_gate(seeds: int = 64) -> dict[str, object]:
    return {
        machine_cls.name: summarize(machine_cls, seeds=seeds)
        for machine_cls in MACHINES
    }


def assert_gate(receipt: dict[str, object]) -> None:
    assert receipt["boundary_phase"]["gate_pass"]
    assert not receipt["whole_window"]["gate_pass"]
    assert not receipt["fixed_window"]["gate_pass"]

    assert (
        receipt["boundary_phase"]["relation_accuracy"]["mean"]
        > 0.82
    )
    assert receipt["boundary_phase"]["boundary_iou"]["mean"] > 0.95
    assert receipt["boundary_phase"]["boundary_mae"]["mean"] < 0.50

    assert receipt["whole_window"]["relation_accuracy"]["mean"] < 0.60
    assert receipt["fixed_window"]["relation_accuracy"]["mean"] < 0.65


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

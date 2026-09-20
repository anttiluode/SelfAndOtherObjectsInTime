"""Gate 3: time creates relational role from one event stream.

Training events are nine samples long. Test events are replayed at 9, 17 and
25 samples with the same normalized temporal structure.

The key question is whether the mechanism represents:
    A -> B
as different from:
    B -> A
when both contain exactly the same identities and object, and whether that
relation survives time stretching.
"""

from __future__ import annotations

import argparse
import json
from typing import Type

import numpy as np

from stream_gate import (
    MACHINES,
    StreamAssociativeMemory,
    StreamWorld,
)


Record = tuple[int, int, int, int, int]


def make_reversal_pairs(
    world: StreamWorld,
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


def run_one(
    machine_cls: Type[StreamAssociativeMemory],
    seed: int,
) -> dict[str, float]:
    world = StreamWorld(seed=seed)
    machine = machine_cls(world)
    records = make_reversal_pairs(world, seed=seed)

    train_rng = np.random.default_rng(seed + 1_000)
    for actor, patient, perspective, object_id, value in records:
        stream = world.event_stream(
            actor,
            patient,
            perspective,
            object_id,
            length=9,
            rng=train_rng,
        )
        machine.write(stream, value)

    result: dict[str, float] = {}

    for length in (9, 17, 25):
        test_rng = np.random.default_rng(seed + 2_000 + length)
        correct = []

        for actor, patient, perspective, object_id, value in records:
            stream = world.event_stream(
                actor,
                patient,
                perspective,
                object_id,
                length=length,
                rng=test_rng,
            )
            correct.append(machine.predict(stream) == value)

        result[f"accuracy_{length}"] = float(np.mean(correct))

    probe_rng = np.random.default_rng(seed + 3_333)
    decomposition_cosines: list[float] = []
    reverse_cosines: list[float] = []

    for actor, patient, perspective, object_id, _ in records[:16]:
        stream = world.event_stream(
            actor,
            patient,
            perspective,
            object_id,
            length=25,
            rng=probe_rng,
        )

        event = machine.parser(stream)
        decomposition_cosines.extend(
            [
                float(event.actor @ world.agents[actor]),
                float(event.patient @ world.agents[patient]),
                float(event.perspective @ world.agents[perspective]),
                float(event.object @ world.objects[object_id]),
            ]
        )

        reversed_event = machine.parser(stream[::-1].copy())
        reverse_cosines.extend(
            [
                float(reversed_event.actor @ world.agents[patient]),
                float(reversed_event.patient @ world.agents[actor]),
                float(
                    reversed_event.perspective
                    @ world.agents[perspective]
                ),
                float(reversed_event.object @ world.objects[object_id]),
            ]
        )

    result["decomposition_cosine"] = float(
        np.mean(decomposition_cosines)
    )
    result["reverse_decode_cosine"] = float(
        np.mean(reverse_cosines)
    )

    return result


def summarize(
    machine_cls: Type[StreamAssociativeMemory],
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
        result["accuracy_9"]["mean"] > 0.85
        and result["accuracy_17"]["mean"] > 0.85
        and result["accuracy_25"]["mean"] > 0.85
        and result["decomposition_cosine"]["mean"] > 0.95
        and result["reverse_decode_cosine"]["mean"] > 0.95
    )
    return result


def run_gate(seeds: int = 64) -> dict[str, object]:
    return {
        machine_cls.name: summarize(machine_cls, seeds=seeds)
        for machine_cls in MACHINES
    }


def assert_gate(receipt: dict[str, object]) -> None:
    assert receipt["phase_relational"]["gate_pass"]
    assert not receipt["symmetric_pair"]["gate_pass"]
    assert not receipt["fixed_clock"]["gate_pass"]

    # Symmetric memory decomposes correctly but throws direction away.
    assert (
        receipt["symmetric_pair"]["decomposition_cosine"]["mean"] > 0.95
    )
    assert receipt["symmetric_pair"]["accuracy_9"]["mean"] < 0.60

    # Fixed clock works at the training duration, then breaks under stretch.
    assert receipt["fixed_clock"]["accuracy_9"]["mean"] > 0.85
    assert receipt["fixed_clock"]["accuracy_17"]["mean"] < 0.70
    assert receipt["fixed_clock"]["accuracy_25"]["mean"] < 0.40


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

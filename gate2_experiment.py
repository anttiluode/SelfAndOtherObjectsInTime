"""Gate 2: distributed role binding without explicit agent IDs."""

from __future__ import annotations

import argparse
import json
from typing import Type

import numpy as np

from relational_gate import (
    MACHINES,
    AssociativeMemory,
    RelationalWorld,
)


Record = tuple[int, int, int, int, int, int]


def make_challenges(
    world: RelationalWorld,
    seed: int,
    pairs_per_group: int = 16,
) -> dict[str, list[Record]]:
    rng = np.random.default_rng(seed + 1_234)
    seen: set[tuple[int, int, int, int, int]] = set()

    groups: dict[str, list[Record]] = {
        "role_swap": [],
        "self_other": [],
        "other_other": [],
    }

    attempts = 0
    while (
        len(groups["role_swap"]) < 2 * pairs_per_group
        and attempts < 200_000
    ):
        attempts += 1
        actor, patient = rng.choice(
            world.n_agents, size=2, replace=False
        )
        perspective = int(rng.integers(world.n_agents))
        object_id = int(rng.integers(world.n_objects))
        tau_id = int(rng.integers(world.n_tau))

        left = (int(actor), int(patient), perspective, object_id, tau_id)
        right = (int(patient), int(actor), perspective, object_id, tau_id)

        if left in seen or right in seen:
            continue

        y_left = int(world.values[left])
        y_right = int(world.values[right])
        if y_left != y_right:
            groups["role_swap"].extend(
                [(*left, y_left), (*right, y_right)]
            )
            seen.update((left, right))

    attempts = 0
    while (
        len(groups["self_other"]) < 2 * pairs_per_group
        and attempts < 200_000
    ):
        attempts += 1
        actor, patient = rng.choice(
            world.n_agents, size=2, replace=False
        )
        other = int(rng.integers(1, world.n_agents))
        object_id = int(rng.integers(world.n_objects))
        tau_id = int(rng.integers(world.n_tau))

        self_view = (int(actor), int(patient), 0, object_id, tau_id)
        other_view = (
            int(actor), int(patient), other, object_id, tau_id
        )

        if self_view in seen or other_view in seen:
            continue

        y_self = int(world.values[self_view])
        y_other = int(world.values[other_view])
        if y_self != y_other:
            groups["self_other"].extend(
                [(*self_view, y_self), (*other_view, y_other)]
            )
            seen.update((self_view, other_view))

    attempts = 0
    while (
        len(groups["other_other"]) < 2 * pairs_per_group
        and attempts < 200_000
    ):
        attempts += 1
        actor, patient = rng.choice(
            world.n_agents, size=2, replace=False
        )
        perspective_a, perspective_b = rng.choice(
            np.arange(1, world.n_agents), size=2, replace=False
        )
        object_id = int(rng.integers(world.n_objects))
        tau_id = int(rng.integers(world.n_tau))

        left = (
            int(actor),
            int(patient),
            int(perspective_a),
            object_id,
            tau_id,
        )
        right = (
            int(actor),
            int(patient),
            int(perspective_b),
            object_id,
            tau_id,
        )

        if left in seen or right in seen:
            continue

        y_left = int(world.values[left])
        y_right = int(world.values[right])
        if y_left != y_right:
            groups["other_other"].extend(
                [(*left, y_left), (*right, y_right)]
            )
            seen.update((left, right))

    expected = 2 * pairs_per_group
    if any(len(records) != expected for records in groups.values()):
        raise RuntimeError("failed to construct complete matched challenges")

    return groups


def run_one(
    machine_cls: Type[AssociativeMemory],
    seed: int,
    pairs_per_group: int = 16,
) -> dict[str, float]:
    world = RelationalWorld(seed=seed)
    machine = machine_cls(world)
    groups = make_challenges(
        world, seed=seed, pairs_per_group=pairs_per_group
    )

    train_rng = np.random.default_rng(seed + 9_999)
    training = [
        record
        for records in groups.values()
        for record in records
    ]
    train_rng.shuffle(training)

    for actor, patient, perspective, object_id, tau_id, value in training:
        observation = world.observation(
            actor, patient, perspective, object_id, tau_id, train_rng
        )
        machine.write(observation, value)

    test_rng = np.random.default_rng(seed + 55_555)
    accuracy: dict[str, float] = {}

    for group_name, records in groups.items():
        correct: list[bool] = []
        for actor, patient, perspective, object_id, tau_id, value in records:
            observation = world.observation(
                actor, patient, perspective, object_id, tau_id, test_rng
            )
            correct.append(machine.predict(observation) == value)
        accuracy[group_name] = float(np.mean(correct))

    return {
        "role_swap_accuracy": accuracy["role_swap"],
        "self_other_accuracy": accuracy["self_other"],
        "other_other_accuracy": accuracy["other_other"],
        "anchor_fidelity": machine.anchor_fidelity,
        "residue_norm": machine.residue_norm,
    }


def summarize(
    machine_cls: Type[AssociativeMemory],
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
        result["role_swap_accuracy"]["mean"] > 0.80
        and result["self_other_accuracy"]["mean"] > 0.70
        and result["other_other_accuracy"]["mean"] > 0.70
        and result["anchor_fidelity"]["mean"] > 0.98
        and result["residue_norm"]["mean"] > 1.0
    )
    return result


def run_gate(seeds: int = 64) -> dict[str, object]:
    return {
        machine_cls.name: summarize(machine_cls, seeds=seeds)
        for machine_cls in MACHINES
    }


def assert_gate(receipt: dict[str, object]) -> None:
    assert receipt["distributed_binder"]["gate_pass"]
    assert not receipt["role_blind_bag"]["gate_pass"]
    assert not receipt["viewless"]["gate_pass"]
    assert not receipt["self_other_binary"]["gate_pass"]
    assert not receipt["mutable_anchor"]["gate_pass"]

    assert (
        receipt["role_blind_bag"]["role_swap_accuracy"]["mean"] < 0.60
    )
    assert receipt["viewless"]["self_other_accuracy"]["mean"] < 0.60
    assert receipt["viewless"]["other_other_accuracy"]["mean"] < 0.60
    assert (
        receipt["self_other_binary"]["other_other_accuracy"]["mean"] < 0.60
    )
    assert receipt["mutable_anchor"]["anchor_fidelity"]["mean"] < 0.90


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

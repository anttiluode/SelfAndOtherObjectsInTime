"""Deterministic Gate 1 receipt.

The same event stream is given to three mechanisms:
- snapshot reset: stable anchor, no post-detour write
- anchored residue: stable anchor + persistent resident write
- leaky overwrite: persistent write, unstable anchor

A gate pass requires both stability and change.
"""

from __future__ import annotations

import argparse
import json
from typing import Type

import numpy as np

from self_other_time import (
    MACHINES,
    PerspectiveMachine,
    TemporalObjectWorld,
)


def make_events(
    world: TemporalObjectWorld, seed: int, n_detours: int
) -> list[tuple[int, int, int, int]]:
    """Create a fixed hidden relation table and a stream of perspective events."""
    rng = np.random.default_rng(seed + 10_000)
    values = rng.choice(
        np.array([-1, 1], dtype=int),
        size=(world.n_agents, world.n_objects, world.n_tau),
    )

    events: list[tuple[int, int, int, int]] = []
    for _ in range(n_detours):
        # Agent zero is the actual self; detours visit only other agents.
        agent_id = int(rng.integers(1, world.n_agents))
        object_id = int(rng.integers(world.n_objects))
        tau_bin = int(rng.integers(world.n_tau))
        value = int(values[agent_id, object_id, tau_bin])
        events.append((agent_id, object_id, tau_bin, value))
    return events


def run_one(
    machine_cls: Type[PerspectiveMachine],
    seed: int,
    n_detours: int = 96,
    recent: int = 20,
) -> dict[str, float]:
    world = TemporalObjectWorld(seed=seed)
    machine = machine_cls(world)
    events = make_events(world, seed, n_detours)

    for event in events:
        machine.detour(*event)

    recall = float(
        np.mean(
            [
                machine.predict(agent_id, object_id, tau_bin) == value
                for agent_id, object_id, tau_bin, value in events[-recent:]
            ]
        )
    )

    return {
        "anchor_fidelity": machine.anchor_fidelity,
        "residue_norm": machine.residue_norm,
        "recent_recall": recall,
    }


def order_sensitivity(
    machine_cls: Type[PerspectiveMachine],
    seed: int,
    n_events: int = 16,
) -> float:
    """Same event multiset, reversed order: does history preserve trajectory?"""
    world = TemporalObjectWorld(seed=seed)
    events = make_events(world, seed, n_events)

    forward = machine_cls(world)
    reverse = machine_cls(world)

    for event in events:
        forward.detour(*event)
    for event in reversed(events):
        reverse.detour(*event)

    denominator = float(
        np.linalg.norm(forward.memory) + np.linalg.norm(reverse.memory) + 1e-12
    )
    return float(np.linalg.norm(forward.memory - reverse.memory) / denominator)


def summarize(
    machine_cls: Type[PerspectiveMachine], seeds: int = 64
) -> dict[str, object]:
    rows = [run_one(machine_cls, seed) for seed in range(seeds)]

    summary: dict[str, object] = {}
    for key in rows[0]:
        values = np.array([row[key] for row in rows], dtype=float)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }

    order = np.array(
        [order_sensitivity(machine_cls, seed) for seed in range(seeds)],
        dtype=float,
    )
    summary["order_sensitivity"] = {
        "mean": float(order.mean()),
        "std": float(order.std()),
    }

    anchor = summary["anchor_fidelity"]["mean"]  # type: ignore[index]
    residue = summary["residue_norm"]["mean"]  # type: ignore[index]
    recall = summary["recent_recall"]["mean"]  # type: ignore[index]
    ordered = summary["order_sensitivity"]["mean"]  # type: ignore[index]

    summary["changed_but_stable"] = bool(
        anchor > 0.98
        and residue > 0.5
        and recall > 0.75
        and ordered > 0.01
    )
    return summary


def run_gate(seeds: int = 64) -> dict[str, object]:
    return {
        machine_cls.name: summarize(machine_cls, seeds=seeds)
        for machine_cls in MACHINES
    }


def assert_gate(receipt: dict[str, object]) -> None:
    reset = receipt["snapshot_reset"]
    anchored = receipt["anchored_residue"]
    overwrite = receipt["leaky_overwrite"]

    assert not reset["changed_but_stable"]
    assert anchored["changed_but_stable"]
    assert not overwrite["changed_but_stable"]

    # Make the two independent failure modes explicit.
    assert reset["anchor_fidelity"]["mean"] > 0.98
    assert reset["residue_norm"]["mean"] == 0.0

    assert overwrite["recent_recall"]["mean"] > 0.75
    assert overwrite["anchor_fidelity"]["mean"] < 0.98


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

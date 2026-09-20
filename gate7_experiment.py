"""Gate 7: learn event-worthiness by active causal probing.

The unknown causal map is one of eight hypotheses. Active and random strategies
receive the same strict budget of three probes.

After probing, both strategies use the same posterior consequence rule to choose
which candidate interruption deserves the child event.
"""

from __future__ import annotations

import argparse
import json
from typing import Type

import numpy as np

from causal_discovery_gate import (
    MACHINES,
    ActiveCausalAdmission,
    CausalAdmissionMemory,
    CausalCandidate,
    ContentWorld,
)


Relation = tuple[int, int, int, int]
Record = tuple[int, Relation, Relation, int, float]


def random_relation(
    world: ContentWorld,
    rng: np.random.Generator,
) -> Relation:
    actor, patient = rng.choice(
        world.n_agents,
        size=2,
        replace=False,
    )

    return (
        int(actor),
        int(patient),
        int(rng.integers(world.n_agents)),
        int(rng.integers(world.n_objects)),
    )


def make_records(
    world: ContentWorld,
    seed: int,
    pairs: int = 20,
) -> list[Record]:
    rng = np.random.default_rng(seed + 11)
    records: list[Record] = []

    for pair_id in range(pairs):
        parent = random_relation(world, rng)
        child_a = random_relation(world, rng)
        child_b = random_relation(world, rng)

        while child_b == child_a:
            child_b = random_relation(world, rng)

        sign = int(rng.choice(np.array([-1, 1])))
        strength = float(rng.uniform(0.35, 0.65))

        records.extend(
            [
                (pair_id, parent, child_a, sign, strength),
                (pair_id, parent, child_b, -sign, strength),
            ]
        )

    return records


def causal_deltas(
    rng: np.random.Generator,
    true_hypothesis: np.ndarray,
    relevant_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    orth = rng.normal(size=3)
    orth -= true_hypothesis * float(true_hypothesis @ orth)
    orth /= np.linalg.norm(orth)

    relevant = (
        relevant_strength * true_hypothesis
        + 0.05 * orth
    )

    distractor_orth = rng.normal(size=3)
    distractor_orth -= (
        true_hypothesis
        * float(true_hypothesis @ distractor_orth)
    )
    distractor_orth /= np.linalg.norm(distractor_orth)

    distractor = (
        float(rng.uniform(2.0, 6.0)) * distractor_orth
        + 0.02 * true_hypothesis
    )

    return relevant, distractor


def make_candidates(
    machine: CausalAdmissionMemory,
    relevant_relation: Relation,
    relevant_strength: float,
    rng: np.random.Generator,
) -> list[CausalCandidate]:
    relevant_delta, distractor_delta = causal_deltas(
        rng,
        machine.true_hypothesis,
        relevant_strength,
    )

    distractor_relation = random_relation(
        machine.world,
        rng,
    )

    candidates = [
        CausalCandidate(
            content=machine.world.observe(
                relevant_relation,
                rng,
            ),
            delta=relevant_delta,
            relevant=True,
        ),
        CausalCandidate(
            content=machine.world.observe(
                distractor_relation,
                rng,
            ),
            delta=distractor_delta,
            relevant=False,
        ),
    ]

    rng.shuffle(candidates)
    return candidates


def run_one(
    machine_cls: Type[CausalAdmissionMemory],
    seed: int,
) -> dict[str, float]:
    world = ContentWorld(seed=seed)
    machine = machine_cls(world, seed=seed)
    records = make_records(world, seed)

    train_rng = np.random.default_rng(seed + 1_000)

    for _, parent_relation, child_relation, value, strength in records:
        parent = world.observe(parent_relation, train_rng)
        candidates = make_candidates(
            machine,
            relevant_relation=child_relation,
            relevant_strength=strength,
            rng=train_rng,
        )
        machine.write(parent, candidates, value)

    test_rng = np.random.default_rng(seed + 2_000)

    correct: list[bool] = []
    relevant_admission: list[bool] = []

    for _, parent_relation, child_relation, value, strength in records:
        parent = world.observe(parent_relation, test_rng)
        candidates = make_candidates(
            machine,
            relevant_relation=child_relation,
            relevant_strength=strength,
            rng=test_rng,
        )

        prediction, admitted_relevant = machine.predict(
            parent,
            candidates,
        )

        correct.append(prediction == value)
        relevant_admission.append(admitted_relevant)

    causal = machine.causal

    return {
        "accuracy": float(np.mean(correct)),
        "relevant_admission": float(
            np.mean(relevant_admission)
        ),
        "hypothesis_accuracy": causal.hypothesis_accuracy,
        "posterior_entropy": causal.posterior_entropy,
        "distinct_probe_categories": (
            causal.distinct_probe_categories
        ),
    }


def summarize(
    machine_cls: Type[CausalAdmissionMemory],
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
        machine_cls is ActiveCausalAdmission
        and result["accuracy"]["mean"] > 0.90
        and result["relevant_admission"]["mean"] > 0.90
        and result["hypothesis_accuracy"]["mean"] > 0.90
        and result["posterior_entropy"]["mean"] < 0.60
        and result["distinct_probe_categories"]["mean"] == 3.0
    )

    return result


def run_gate(seeds: int = 64) -> dict[str, object]:
    return {
        machine_cls.name: summarize(machine_cls, seeds=seeds)
        for machine_cls in MACHINES
    }


def assert_gate(receipt: dict[str, object]) -> None:
    assert receipt["active_eig"]["gate_pass"]
    assert not receipt["oracle"]["gate_pass"]
    assert not receipt["random_probe"]["gate_pass"]
    assert not receipt["no_probe"]["gate_pass"]

    active = receipt["active_eig"]
    random = receipt["random_probe"]
    none = receipt["no_probe"]

    assert active["accuracy"]["mean"] > 0.90
    assert active["relevant_admission"]["mean"] > 0.90
    assert active["hypothesis_accuracy"]["mean"] > 0.90

    assert random["accuracy"]["mean"] < 0.80
    assert random["hypothesis_accuracy"]["mean"] < 0.70
    assert (
        random["posterior_entropy"]["mean"]
        > active["posterior_entropy"]["mean"]
    )

    assert none["accuracy"]["mean"] < 0.60
    assert none["posterior_entropy"]["mean"] > 2.9


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

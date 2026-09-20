"""Gate 6: which interruption deserves its own event?"""

from __future__ import annotations

import argparse
import json
from typing import Type

import numpy as np

from admission_gate import (
    MACHINES,
    AdmissionMemory,
    AdmissionWorld,
    Candidate,
    ConsequenceAdmission,
    Relation,
)


Record = tuple[int, Relation, Relation, int, float]


def random_relation(
    world: AdmissionWorld,
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
    world: AdmissionWorld,
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


def make_candidates(
    world: AdmissionWorld,
    relevant_relation: Relation,
    relevant_strength: float,
    rng: np.random.Generator,
) -> tuple[list[Candidate], Candidate, Candidate]:
    relevant = Candidate(
        content=world.observe_relation(relevant_relation, rng),
        delta=world.candidate_delta(
            relevant=True,
            rng=rng,
            strength=relevant_strength,
        ),
        relevant=True,
    )

    distractor_relation = random_relation(world, rng)

    distractor = Candidate(
        content=world.observe_relation(distractor_relation, rng),
        delta=world.candidate_delta(
            relevant=False,
            rng=rng,
        ),
        relevant=False,
    )

    candidates = [relevant, distractor]
    rng.shuffle(candidates)

    return candidates, relevant, distractor


def run_one(
    machine_cls: Type[AdmissionMemory],
    seed: int,
) -> dict[str, float]:
    world = AdmissionWorld(seed=seed)
    machine = machine_cls(world)
    records = make_records(world, seed)

    train_rng = np.random.default_rng(seed + 1_000)

    for _, parent_relation, child_relation, value, strength in records:
        parent = world.observe_relation(parent_relation, train_rng)
        candidates, _, _ = make_candidates(
            world,
            relevant_relation=child_relation,
            relevant_strength=strength,
            rng=train_rng,
        )
        machine.write(parent, candidates, value)

    test_rng = np.random.default_rng(seed + 2_000)

    correct: list[bool] = []
    precision: list[float] = []
    raw_ratios: list[float] = []
    consequence_ratios: list[float] = []
    pair_scores: dict[int, list[tuple[float, int]]] = {}

    for (
        pair_id,
        parent_relation,
        child_relation,
        value,
        strength,
    ) in records:
        parent = world.observe_relation(parent_relation, test_rng)
        candidates, relevant, distractor = make_candidates(
            world,
            relevant_relation=child_relation,
            relevant_strength=strength,
            rng=test_rng,
        )

        correct.append(machine.predict(parent, candidates) == value)
        precision.append(machine.admission_precision(candidates))

        relevant_score = world.consequence_score(relevant)
        distractor_score = world.consequence_score(distractor)

        pair_scores.setdefault(pair_id, []).append(
            (relevant_score, value)
        )

        raw_ratios.append(
            float(
                np.linalg.norm(distractor.delta)
                / np.linalg.norm(relevant.delta)
            )
        )

        consequence_ratios.append(
            float(
                relevant_score
                / (distractor_score + 1e-12)
            )
        )

    matched_score_gap = np.mean(
        [
            abs(entries[0][0] - entries[1][0])
            for entries in pair_scores.values()
        ]
    )

    return {
        "accuracy": float(np.mean(correct)),
        "admission_precision": float(np.mean(precision)),
        "matched_score_gap": float(matched_score_gap),
        "raw_distractor_to_relevant_ratio": float(
            np.mean(raw_ratios)
        ),
        "relevant_to_distractor_consequence_ratio": float(
            np.mean(consequence_ratios)
        ),
    }


def summarize(
    machine_cls: Type[AdmissionMemory],
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
        machine_cls is ConsequenceAdmission
        and result["accuracy"]["mean"] > 0.95
        and result["admission_precision"]["mean"] > 0.95
        and result["matched_score_gap"]["mean"] < 1e-12
        and result[
            "raw_distractor_to_relevant_ratio"
        ]["mean"] > 5.0
        and result[
            "relevant_to_distractor_consequence_ratio"
        ]["mean"] > 15.0
    )

    return result


def run_gate(seeds: int = 64) -> dict[str, object]:
    return {
        machine_cls.name: summarize(machine_cls, seeds=seeds)
        for machine_cls in MACHINES
    }


def assert_gate(receipt: dict[str, object]) -> None:
    assert receipt["consequence"]["gate_pass"]
    assert not receipt["magnitude"]["gate_pass"]
    assert not receipt["admit_all"]["gate_pass"]
    assert not receipt["ignore"]["gate_pass"]

    assert receipt["consequence"]["accuracy"]["mean"] > 0.95
    assert (
        receipt["consequence"]["admission_precision"]["mean"]
        > 0.95
    )

    assert receipt["magnitude"]["accuracy"]["mean"] < 0.60
    assert receipt["admit_all"]["accuracy"]["mean"] < 0.65
    assert receipt["ignore"]["accuracy"]["mean"] < 0.60

    assert (
        receipt["consequence"][
            "raw_distractor_to_relevant_ratio"
        ]["mean"]
        > 5.0
    )
    assert (
        receipt["consequence"][
            "relevant_to_distractor_consequence_ratio"
        ]["mean"]
        > 15.0
    )
    assert (
        receipt["consequence"]["matched_score_gap"]["mean"]
        < 1e-12
    )


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

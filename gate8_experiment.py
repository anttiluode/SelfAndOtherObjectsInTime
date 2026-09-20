"""Gate 8: allocate temporal precision according to consequence.

Both events are already admitted. The task label is the fine temporal order in
the consequential event.

The consequential slot alternates across trials. The low-consequence event has
larger raw amplitude, preventing a magnitude heuristic from serving as a proxy.

All budget-respecting strategies spend exactly 16 bins total. A fine-everywhere
control spends 24 and therefore cannot pass despite perfect accuracy.
"""

from __future__ import annotations

import argparse
import json
from typing import Type

import numpy as np

from temporal_allocation_gate import (
    MICRO_PAIRS,
    POLICIES,
    AllocationPolicy,
    ConsequenceAdaptive,
    TimedEvent,
    TOTAL_BUDGET,
    decode_order,
)


def make_trial(
    rng: np.random.Generator,
    relevant_slot: int,
) -> list[TimedEvent]:
    events: list[TimedEvent] = []

    for slot in range(2):
        relevant = slot == relevant_slot

        events.append(
            TimedEvent(
                order=int(rng.choice(np.array([-1, 1]))),
                micro_pair=MICRO_PAIRS[
                    int(rng.integers(len(MICRO_PAIRS)))
                ],
                consequence=1.0 if relevant else 0.2,
                amplitude=(
                    1.0
                    if relevant
                    else float(rng.uniform(3.0, 5.0))
                ),
                relevant=relevant,
            )
        )

    return events


def random_guess(
    rng: np.random.Generator,
) -> int:
    return int(rng.choice(np.array([-1, 1])))


def run_one(
    policy_cls: Type[AllocationPolicy],
    seed: int,
    trials: int = 200,
) -> dict[str, float]:
    rng = np.random.default_rng(seed + 8_000)
    policy = policy_cls()

    correct: list[bool] = []
    irrelevant_correct: list[bool] = []
    relevant_resolution: list[float] = []
    irrelevant_resolution: list[float] = []
    resource_cost: list[float] = []
    swapped_slot_correct: list[bool] = []

    for trial in range(trials):
        # Relevance moves between slots. A fixed location prior cannot solve
        # the held-out stream.
        relevant_slot = trial % 2
        irrelevant_slot = 1 - relevant_slot

        events = make_trial(
            rng,
            relevant_slot=relevant_slot,
        )
        allocation = policy.allocate(events)

        prediction = decode_order(
            events[relevant_slot],
            allocation[relevant_slot],
        )

        if prediction == 0:
            prediction = random_guess(rng)

        is_correct = (
            prediction == events[relevant_slot].order
        )
        correct.append(is_correct)

        irrelevant_prediction = decode_order(
            events[irrelevant_slot],
            allocation[irrelevant_slot],
        )

        if irrelevant_prediction == 0:
            irrelevant_prediction = random_guess(rng)

        irrelevant_correct.append(
            irrelevant_prediction
            == events[irrelevant_slot].order
        )

        relevant_resolution.append(
            allocation[relevant_slot]
        )
        irrelevant_resolution.append(
            allocation[irrelevant_slot]
        )
        resource_cost.append(sum(allocation))

        if relevant_slot == 1:
            swapped_slot_correct.append(is_correct)

    return {
        "accuracy": float(np.mean(correct)),
        "swapped_slot_accuracy": float(
            np.mean(swapped_slot_correct)
        ),
        "relevant_resolution": float(
            np.mean(relevant_resolution)
        ),
        "irrelevant_resolution": float(
            np.mean(irrelevant_resolution)
        ),
        "irrelevant_order_accuracy": float(
            np.mean(irrelevant_correct)
        ),
        "resource_cost": float(
            np.mean(resource_cost)
        ),
    }


def summarize(
    policy_cls: Type[AllocationPolicy],
    seeds: int = 64,
) -> dict[str, object]:
    rows = [
        run_one(policy_cls, seed)
        for seed in range(seeds)
    ]

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
        policy_cls is ConsequenceAdaptive
        and result["accuracy"]["mean"] > 0.99
        and result["swapped_slot_accuracy"]["mean"] > 0.99
        and result["resource_cost"]["mean"] <= TOTAL_BUDGET
        and result["relevant_resolution"]["mean"] == 12.0
        and result["irrelevant_resolution"]["mean"] == 4.0
        and result["irrelevant_order_accuracy"]["mean"] < 0.58
    )

    return result


def run_gate(seeds: int = 64) -> dict[str, object]:
    return {
        policy_cls.name: summarize(
            policy_cls,
            seeds=seeds,
        )
        for policy_cls in POLICIES
    }


def assert_gate(
    receipt: dict[str, object],
) -> None:
    adaptive = receipt["consequence_adaptive"]
    uniform = receipt["uniform"]
    magnitude = receipt["magnitude"]
    fixed = receipt["fixed_slot"]
    fine = receipt["fine_everywhere"]

    assert adaptive["gate_pass"]
    assert not uniform["gate_pass"]
    assert not magnitude["gate_pass"]
    assert not fixed["gate_pass"]
    assert not fine["gate_pass"]

    assert adaptive["accuracy"]["mean"] > 0.99
    assert adaptive["swapped_slot_accuracy"]["mean"] > 0.99
    assert adaptive["resource_cost"]["mean"] == TOTAL_BUDGET

    assert uniform["accuracy"]["mean"] < 0.58
    assert magnitude["accuracy"]["mean"] < 0.58

    assert fixed["accuracy"]["mean"] < 0.80
    assert fixed["swapped_slot_accuracy"]["mean"] < 0.60

    # Fine timing everywhere works, but only by spending 50% more resource.
    assert fine["accuracy"]["mean"] > 0.99
    assert fine["resource_cost"]["mean"] > TOTAL_BUDGET


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--assert-gate", action="store_true")
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    receipt = run_gate(seeds=args.seeds)

    if args.assert_gate:
        assert_gate(receipt)

    text = json.dumps(
        receipt,
        indent=2,
        sort_keys=True,
    )
    print(text)

    if args.json:
        with open(
            args.json,
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()

"""Gate 9: timing bandwidth moves without resetting the event.

Each episode contains the same two continuing event identities across four
epochs. Consequence alternates between them every epoch. The target is the fine
temporal order in the currently consequential stream, interpreted through a
resident context sign established only once at event start.

Epoch durations are independently warped by x1/x2/x3. All budget-respecting
strategies spend exactly 16 timing bins per epoch.
"""

from __future__ import annotations

import argparse
import json
from typing import Type

import numpy as np

from dynamic_temporal_gate import (
    MICRO_PAIRS,
    POLICIES,
    DynamicConsequencePolicy,
    EpochEvent,
    OnlinePolicy,
    TOTAL_BUDGET,
    decode_order,
)


def make_epoch_events(
    rng: np.random.Generator,
    relevant_slot: int,
    warp: int,
) -> list[EpochEvent]:
    events: list[EpochEvent] = []

    for slot in range(2):
        relevant = slot == relevant_slot

        events.append(
            EpochEvent(
                order=int(rng.choice(np.array([-1, 1]))),
                micro_pair=MICRO_PAIRS[
                    int(rng.integers(len(MICRO_PAIRS)))
                ],
                warp=warp,
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


def random_sign(
    rng: np.random.Generator,
) -> int:
    return int(rng.choice(np.array([-1, 1])))


def run_one(
    policy_cls: Type[OnlinePolicy],
    seed: int,
    episodes: int = 100,
) -> dict[str, float]:
    rng = np.random.default_rng(seed + 9_000)
    policy = policy_cls()

    correct: list[bool] = []
    warped_correct: list[bool] = []
    post_switch_correct: list[bool] = []
    resource_cost: list[float] = []

    for _ in range(episodes):
        context_sign = random_sign(rng)
        initial_relevant_slot = int(rng.integers(2))

        schedule = (
            initial_relevant_slot,
            1 - initial_relevant_slot,
            initial_relevant_slot,
            1 - initial_relevant_slot,
        )

        policy.begin_episode(
            initial_relevant_slot=initial_relevant_slot,
            context_sign=context_sign,
        )

        for epoch, relevant_slot in enumerate(schedule):
            warp = int(rng.choice(np.array([1, 2, 3])))

            events = make_epoch_events(
                rng,
                relevant_slot=relevant_slot,
                warp=warp,
            )

            allocation = policy.allocation(
                events,
                epoch=epoch,
                current_relevant_slot=relevant_slot,
            )

            decoded = decode_order(
                events[relevant_slot],
                allocation[relevant_slot],
            )

            context_used = policy.context_for_epoch(
                rng,
                epoch=epoch,
                current_relevant_slot=relevant_slot,
            )

            target = (
                context_sign
                * events[relevant_slot].order
            )

            if decoded == 0:
                prediction = random_sign(rng)
            else:
                prediction = context_used * decoded

            is_correct = prediction == target
            correct.append(is_correct)

            if warp > 1:
                warped_correct.append(is_correct)

            if epoch > 0:
                post_switch_correct.append(is_correct)

            resource_cost.append(sum(allocation))

            policy.finish_epoch(relevant_slot)

    return {
        "accuracy": float(np.mean(correct)),
        "warped_accuracy": float(
            np.mean(warped_correct)
        ),
        "post_switch_accuracy": float(
            np.mean(post_switch_correct)
        ),
        "resource_cost": float(
            np.mean(resource_cost)
        ),
    }


def summarize(
    policy_cls: Type[OnlinePolicy],
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
        policy_cls is DynamicConsequencePolicy
        and result["accuracy"]["mean"] > 0.99
        and result["warped_accuracy"]["mean"] > 0.99
        and result["post_switch_accuracy"]["mean"] > 0.99
        and result["resource_cost"]["mean"] <= TOTAL_BUDGET
    )

    return result


def run_gate(
    seeds: int = 64,
) -> dict[str, object]:
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
    dynamic = receipt["dynamic_consequence"]
    static = receipt["static_start"]
    lag = receipt["lag_one"]
    uniform = receipt["uniform"]
    magnitude = receipt["magnitude"]
    reset = receipt["reset_on_reallocation"]
    fine = receipt["fine_everywhere"]

    assert dynamic["gate_pass"]
    assert dynamic["accuracy"]["mean"] > 0.99
    assert dynamic["warped_accuracy"]["mean"] > 0.99
    assert dynamic["post_switch_accuracy"]["mean"] > 0.99
    assert dynamic["resource_cost"]["mean"] == TOTAL_BUDGET

    assert static["accuracy"]["mean"] < 0.80
    assert static["post_switch_accuracy"]["mean"] < 0.72

    assert lag["accuracy"]["mean"] < 0.70
    assert lag["post_switch_accuracy"]["mean"] < 0.58

    assert uniform["accuracy"]["mean"] < 0.58
    assert magnitude["accuracy"]["mean"] < 0.58

    # Correct reallocation is not enough if it resets the resident event.
    assert reset["accuracy"]["mean"] < 0.70
    assert reset["post_switch_accuracy"]["mean"] < 0.58

    # Fine timing everywhere remains an over-budget performance control.
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

"""Gate 7: active causal identification for event admission.

Gate 6 received the downstream consequence map. Gate 7 hides it.

The world has one of eight possible downstream-sensitivity hypotheses. A strict
three-probe budget is shared by all non-oracle strategies. Each probe returns a
noisy binary consequence observation. Expected information gain chooses probes
that split the remaining hypothesis mass rather than repeating redundant tests.

After probing, event admission uses posterior expected consequence:
    E_h[ |w_h^T delta| ]

The hypothesis family and observation likelihood are still supplied. This gate
tests active identification of which causal map is present, not discovery of an
arbitrary nonlinear causal model.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np

from admission_gate import unit


HYPOTHESES = np.asarray(
    list(itertools.product((-1.0, 1.0), repeat=3)),
    dtype=float,
) / np.sqrt(3.0)

_probe_list: list[np.ndarray] = []

for axis in range(3):
    for _ in range(3):
        probe = np.zeros(3, dtype=float)
        probe[axis] = 1.0
        _probe_list.append(probe)

for _ in range(3):
    _probe_list.append(
        np.ones(3, dtype=float) / np.sqrt(3.0)
    )

PROBES = np.stack(_probe_list)

FLIP_PROBABILITY = 0.02
PROBE_BUDGET = 3


def likelihood_table() -> np.ndarray:
    table = np.zeros(
        (len(PROBES), len(HYPOTHESES), 2),
        dtype=float,
    )

    for probe_index, probe in enumerate(PROBES):
        for hypothesis_index, hypothesis in enumerate(HYPOTHESES):
            sign = 1 if float(hypothesis @ probe) >= 0.0 else -1
            correct_index = 1 if sign == 1 else 0
            wrong_index = 1 - correct_index

            table[
                probe_index,
                hypothesis_index,
                correct_index,
            ] = 1.0 - FLIP_PROBABILITY
            table[
                probe_index,
                hypothesis_index,
                wrong_index,
            ] = FLIP_PROBABILITY

    return table


LIKELIHOOD = likelihood_table()


def entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0.0]

    return float(
        -np.sum(positive * np.log2(positive))
    )


def expected_information_gain(
    posterior: np.ndarray,
    probe_index: int,
) -> float:
    prior_entropy = entropy(posterior)
    expected_after = 0.0

    for outcome_index in (0, 1):
        likelihood = LIKELIHOOD[
            probe_index,
            :,
            outcome_index,
        ]
        probability = float(posterior @ likelihood)

        if probability == 0.0:
            continue

        updated = posterior * likelihood / probability
        expected_after += probability * entropy(updated)

    return prior_entropy - expected_after


def choose_active_probe(
    posterior: np.ndarray,
    used: list[int],
) -> int:
    best_gain = -np.inf
    best_index = -1

    for probe_index in range(len(PROBES)):
        if probe_index in used:
            continue

        gain = expected_information_gain(
            posterior,
            probe_index,
        )

        if gain > best_gain + 1e-12:
            best_gain = gain
            best_index = probe_index

    if best_index < 0:
        raise RuntimeError("no probe available")

    return best_index


def probe_category(probe_index: int) -> int:
    if probe_index < 9:
        return probe_index // 3

    return 3


@dataclass
class CausalPosterior:
    true_index: int
    posterior: np.ndarray
    used_probes: list[int]

    @property
    def hypothesis_accuracy(self) -> float:
        return float(
            int(np.argmax(self.posterior) == self.true_index)
        )

    @property
    def posterior_entropy(self) -> float:
        return entropy(self.posterior)

    @property
    def distinct_probe_categories(self) -> float:
        return float(
            len(
                {
                    probe_category(probe)
                    for probe in self.used_probes
                }
            )
        )


def infer_causal_map(
    seed: int,
    strategy: str,
) -> CausalPosterior:
    rng = np.random.default_rng(seed + 7_000)
    true_index = int(rng.integers(len(HYPOTHESES)))

    if strategy == "oracle":
        posterior = np.zeros(len(HYPOTHESES), dtype=float)
        posterior[true_index] = 1.0

        return CausalPosterior(
            true_index=true_index,
            posterior=posterior,
            used_probes=[],
        )

    posterior = np.ones(
        len(HYPOTHESES),
        dtype=float,
    ) / len(HYPOTHESES)

    used: list[int] = []

    for _ in range(PROBE_BUDGET):
        if strategy == "none":
            break

        if strategy == "active":
            probe_index = choose_active_probe(
                posterior,
                used,
            )
        elif strategy == "random":
            available = [
                index
                for index in range(len(PROBES))
                if index not in used
            ]
            probe_index = int(rng.choice(available))
        else:
            raise ValueError(f"unknown strategy: {strategy}")

        used.append(probe_index)

        true_hypothesis = HYPOTHESES[true_index]
        sign = (
            1
            if float(true_hypothesis @ PROBES[probe_index]) >= 0.0
            else -1
        )

        if float(rng.random()) < FLIP_PROBABILITY:
            sign *= -1

        outcome_index = 1 if sign == 1 else 0

        posterior *= LIKELIHOOD[
            probe_index,
            :,
            outcome_index,
        ]
        posterior /= posterior.sum()

    return CausalPosterior(
        true_index=true_index,
        posterior=posterior,
        used_probes=used,
    )


class ContentWorld:
    """Distributed relation addresses independent of the hidden causal map."""

    def __init__(
        self,
        seed: int,
        dim: int = 96,
        cue_noise: float = 0.03,
    ) -> None:
        rng = np.random.default_rng(seed)

        self.dim = dim
        self.cue_noise = cue_noise
        self.n_agents = 8
        self.n_objects = 12

        self.agents = np.stack(
            [unit(rng.normal(size=dim)) for _ in range(self.n_agents)]
        )
        self.objects = np.stack(
            [unit(rng.normal(size=dim)) for _ in range(self.n_objects)]
        )

        self.roles = {
            name: rng.choice(np.array([-1.0, 1.0]), size=dim)
            for name in ("actor", "patient", "view", "object", "child")
        }

    def relation_key(
        self,
        relation: tuple[int, int, int, int],
    ) -> np.ndarray:
        actor, patient, perspective, object_id = relation
        r = self.roles

        return unit(
            r["actor"] * self.agents[actor]
            + r["patient"] * self.agents[patient]
            + r["view"] * self.agents[perspective]
            + 0.8 * r["object"] * self.objects[object_id]
        )

    def observe(
        self,
        relation: tuple[int, int, int, int],
        rng: np.random.Generator,
    ) -> np.ndarray:
        return unit(
            self.relation_key(relation)
            + self.cue_noise * rng.normal(size=self.dim)
        )


@dataclass
class CausalCandidate:
    content: np.ndarray
    delta: np.ndarray
    relevant: bool


class CausalAdmissionMemory:
    name = "base"
    strategy = "none"

    def __init__(
        self,
        content_world: ContentWorld,
        seed: int,
    ) -> None:
        self.world = content_world
        self.causal = infer_causal_map(seed, self.strategy)
        self.keys: list[np.ndarray] = []
        self.labels: list[float] = []

    @property
    def true_hypothesis(self) -> np.ndarray:
        return HYPOTHESES[self.causal.true_index]

    def consequence_score(
        self,
        candidate: CausalCandidate,
    ) -> float:
        return float(
            np.sum(
                self.causal.posterior
                * np.abs(HYPOTHESES @ candidate.delta)
            )
        )

    def choose(
        self,
        candidates: list[CausalCandidate],
    ) -> int:
        scores = [
            self.consequence_score(candidate)
            for candidate in candidates
        ]
        return int(np.argmax(scores))

    def compose(
        self,
        parent: np.ndarray,
        candidates: list[CausalCandidate],
    ) -> tuple[np.ndarray, bool]:
        index = self.choose(candidates)
        candidate = candidates[index]

        key = unit(
            parent
            + 0.8
            * self.world.roles["child"]
            * candidate.content
        )

        return key, candidate.relevant

    def write(
        self,
        parent: np.ndarray,
        candidates: list[CausalCandidate],
        value: int,
    ) -> None:
        key, _ = self.compose(parent, candidates)
        self.keys.append(key)
        self.labels.append(float(value))

    def predict(
        self,
        parent: np.ndarray,
        candidates: list[CausalCandidate],
    ) -> tuple[int, bool]:
        query, relevant = self.compose(parent, candidates)
        keys = np.stack(self.keys)
        similarities = keys @ query

        score = float(
            np.dot(
                np.asarray(self.labels),
                similarities * similarities,
            )
        )

        return (1 if score >= 0.0 else -1), relevant


class OracleCausalAdmission(CausalAdmissionMemory):
    name = "oracle"
    strategy = "oracle"


class ActiveCausalAdmission(CausalAdmissionMemory):
    name = "active_eig"
    strategy = "active"


class RandomProbeAdmission(CausalAdmissionMemory):
    name = "random_probe"
    strategy = "random"


class NoProbeAdmission(CausalAdmissionMemory):
    name = "no_probe"
    strategy = "none"


MACHINES = (
    OracleCausalAdmission,
    ActiveCausalAdmission,
    RandomProbeAdmission,
    NoProbeAdmission,
)

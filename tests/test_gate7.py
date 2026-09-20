import numpy as np

from causal_discovery_gate import (
    ActiveCausalAdmission,
    ContentWorld,
    HYPOTHESES,
    PROBE_BUDGET,
    infer_causal_map,
)
from gate7_experiment import assert_gate, run_gate


def test_active_uses_three_nonredundant_probe_categories():
    posterior = infer_causal_map(seed=2, strategy="active")

    assert len(posterior.used_probes) == PROBE_BUDGET
    assert posterior.distinct_probe_categories == 3.0


def test_active_posterior_reduces_uncertainty():
    posterior = infer_causal_map(seed=3, strategy="active")
    none = infer_causal_map(seed=3, strategy="none")

    assert posterior.posterior_entropy < none.posterior_entropy
    assert none.posterior_entropy == 3.0


def test_candidate_scoring_uses_posterior_expected_consequence():
    world = ContentWorld(seed=4)
    machine = ActiveCausalAdmission(world, seed=4)

    true_hypothesis = HYPOTHESES[machine.causal.true_index]

    relevant = 0.5 * true_hypothesis

    orth = np.array([1.0, 0.0, 0.0])
    orth -= true_hypothesis * float(true_hypothesis @ orth)

    if np.linalg.norm(orth) < 1e-8:
        orth = np.array([0.0, 1.0, 0.0])
        orth -= true_hypothesis * float(true_hypothesis @ orth)

    orth /= np.linalg.norm(orth)
    distractor = 4.0 * orth + 0.02 * true_hypothesis

    relevant_score = float(
        np.sum(
            machine.causal.posterior
            * np.abs(HYPOTHESES @ relevant)
        )
    )
    distractor_score = float(
        np.sum(
            machine.causal.posterior
            * np.abs(HYPOTHESES @ distractor)
        )
    )

    assert relevant_score > distractor_score


def test_gate7_receipt():
    receipt = run_gate(seeds=16)
    assert_gate(receipt)

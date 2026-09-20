import numpy as np

from admission_gate import (
    AdmitAll,
    AdmissionWorld,
    ConsequenceAdmission,
    IgnoreInterruptions,
    MagnitudeAdmission,
)
from gate6_experiment import (
    assert_gate,
    make_candidates,
    random_relation,
    run_gate,
)


def test_consequence_and_magnitude_choose_opposite_candidates():
    world = AdmissionWorld(seed=1)
    rng = np.random.default_rng(20)
    relation = random_relation(world, rng)

    candidates, relevant, distractor = make_candidates(
        world,
        relevant_relation=relation,
        relevant_strength=0.5,
        rng=rng,
    )

    consequence = ConsequenceAdmission(world)
    magnitude = MagnitudeAdmission(world)

    consequence_index = consequence.admitted(candidates)[0][0]
    magnitude_index = magnitude.admitted(candidates)[0][0]

    assert candidates[consequence_index].relevant
    assert not candidates[magnitude_index].relevant

    assert np.linalg.norm(distractor.delta) > np.linalg.norm(
        relevant.delta
    )
    assert (
        world.consequence_score(relevant)
        > 10.0 * world.consequence_score(distractor)
    )


def test_matched_opposite_labels_can_have_identical_admission_score():
    world = AdmissionWorld(seed=2)
    rng = np.random.default_rng(30)

    relation_a = random_relation(world, rng)
    relation_b = random_relation(world, rng)
    strength = 0.47

    _, relevant_a, _ = make_candidates(
        world,
        relevant_relation=relation_a,
        relevant_strength=strength,
        rng=rng,
    )
    _, relevant_b, _ = make_candidates(
        world,
        relevant_relation=relation_b,
        relevant_strength=strength,
        rng=rng,
    )

    assert not np.allclose(relevant_a.content, relevant_b.content)
    assert abs(
        world.consequence_score(relevant_a)
        - world.consequence_score(relevant_b)
    ) < 1e-12


def test_admit_all_has_only_half_precision_with_one_relevant_child():
    world = AdmissionWorld(seed=3)
    rng = np.random.default_rng(40)
    relation = random_relation(world, rng)

    candidates, _, _ = make_candidates(
        world,
        relevant_relation=relation,
        relevant_strength=0.5,
        rng=rng,
    )

    assert AdmitAll(world).admission_precision(candidates) == 0.5
    assert IgnoreInterruptions(world).admission_precision(candidates) == 0.0


def test_gate6_receipt():
    receipt = run_gate(seeds=16)
    assert_gate(receipt)

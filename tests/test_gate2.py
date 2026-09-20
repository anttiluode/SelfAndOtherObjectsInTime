import numpy as np

from gate2_experiment import assert_gate, run_gate
from relational_gate import (
    DistributedRoleBinder,
    MutableAnchorBinder,
    RelationalWorld,
    RoleBlindBag,
    SelfOtherBinaryMemory,
    ViewlessMemory,
)


def clean_observation(world, actor, patient, perspective, object_id, tau_id):
    rng = np.random.default_rng(123)
    old_noise = world.cue_noise
    world.cue_noise = 0.0
    try:
        return world.observation(
            actor, patient, perspective, object_id, tau_id, rng
        )
    finally:
        world.cue_noise = old_noise


def test_role_binding_is_directional_without_agent_ids():
    world = RelationalWorld(seed=1)
    binder = DistributedRoleBinder(world)
    bag = RoleBlindBag(world)

    ab = clean_observation(world, 1, 2, 0, 3, 1)
    ba = clean_observation(world, 2, 1, 0, 3, 1)

    assert float(binder.key(ab) @ binder.key(ba)) < 0.95
    assert np.allclose(bag.key(ab), bag.key(ba))


def test_perspective_is_part_of_the_address():
    world = RelationalWorld(seed=2)
    binder = DistributedRoleBinder(world)
    viewless = ViewlessMemory(world)

    self_view = clean_observation(world, 1, 2, 0, 4, 2)
    other_view = clean_observation(world, 1, 2, 3, 4, 2)

    assert float(binder.key(self_view) @ binder.key(other_view)) < 0.95
    assert np.allclose(viewless.key(self_view), viewless.key(other_view))


def test_binary_self_other_is_not_enough_for_multiple_others():
    world = RelationalWorld(seed=4)
    binary = SelfOtherBinaryMemory(world)

    other_a = clean_observation(world, 1, 2, 3, 4, 2)
    other_b = clean_observation(world, 1, 2, 5, 4, 2)

    assert np.allclose(binary.key(other_a), binary.key(other_b))


def test_mutable_anchor_drifts_after_many_other_perspectives():
    world = RelationalWorld(seed=5)
    machine = MutableAnchorBinder(world)
    rng = np.random.default_rng(99)

    for i in range(80):
        perspective = 1 + (i % (world.n_agents - 1))
        observation = world.observation(
            actor_id=i % world.n_agents,
            patient_id=(i + 1) % world.n_agents,
            perspective_id=perspective,
            object_id=i % world.n_objects,
            tau_id=i % world.n_tau,
            rng=rng,
        )
        machine.write(observation, 1 if i % 2 == 0 else -1)

    assert machine.anchor_fidelity < 0.90


def test_gate2_receipt():
    receipt = run_gate(seeds=16)
    assert_gate(receipt)

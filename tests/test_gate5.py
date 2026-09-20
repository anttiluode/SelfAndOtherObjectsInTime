import numpy as np

from gate5_experiment import assert_gate, run_gate
from nested_gate import (
    FlatTimelineMemory,
    NestedStackMemory,
    NestedWorld,
    NoWritebackMemory,
    ResetOnResumeMemory,
)


def make_episode(
    world,
    parent_length,
    child_length,
    child_insert,
    seed=123,
):
    rng = np.random.default_rng(seed)

    return world.nested_episode(
        parent=(1, 5, 0, 2),
        child=(3, 6, 4, 7),
        parent_length=parent_length,
        child_length=child_length,
        child_insert=child_insert,
        rng=rng,
    )


def test_nested_stack_is_invariant_to_child_duration():
    world = NestedWorld(seed=1, noise=0.0)
    memory = NestedStackMemory(world)

    short = make_episode(
        world,
        parent_length=17,
        child_length=7,
        child_insert=8,
        seed=20,
    )
    long = make_episode(
        world,
        parent_length=23,
        child_length=29,
        child_insert=11,
        seed=21,
    )

    assert float(memory.key(short) @ memory.key(long)) > 0.96


def test_flat_timeline_changes_when_child_occupies_more_global_time():
    world = NestedWorld(seed=2, noise=0.0)
    memory = FlatTimelineMemory(world)

    short = make_episode(
        world,
        parent_length=17,
        child_length=7,
        child_insert=8,
        seed=30,
    )
    long = make_episode(
        world,
        parent_length=23,
        child_length=29,
        child_insert=11,
        seed=31,
    )

    assert float(memory.key(short) @ memory.key(long)) < 0.70


def test_no_writeback_collapses_two_different_children():
    world = NestedWorld(seed=3, noise=0.0)
    memory = NoWritebackMemory(world)
    rng = np.random.default_rng(40)

    first = world.nested_episode(
        parent=(1, 5, 0, 2),
        child=(3, 6, 4, 7),
        parent_length=19,
        child_length=15,
        child_insert=9,
        rng=rng,
    )
    second = world.nested_episode(
        parent=(1, 5, 0, 2),
        child=(2, 7, 1, 4),
        parent_length=19,
        child_length=15,
        child_insert=9,
        rng=rng,
    )

    assert float(memory.key(first) @ memory.key(second)) > 0.99


def test_reset_on_resume_loses_pre_interruption_parent():
    world = NestedWorld(seed=4, noise=0.0)
    nested = NestedStackMemory(world)
    reset = ResetOnResumeMemory(world)

    tokens = make_episode(
        world,
        parent_length=17,
        child_length=11,
        child_insert=8,
        seed=50,
    )

    # Both mechanisms can return a key, but the reset key is not the same
    # resumed-parent state as the nested mechanism.
    assert float(nested.key(tokens) @ reset.key(tokens)) < 0.80


def test_gate5_receipt():
    receipt = run_gate(seeds=16)
    assert_gate(receipt)

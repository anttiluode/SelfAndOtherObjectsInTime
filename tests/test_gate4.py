import numpy as np

from boundary_gate import (
    BoundaryPhaseMemory,
    ContinuousWorld,
    FixedWindowMemory,
    WholeWindowMemory,
    discover_boundaries,
)
from gate4_experiment import assert_gate, interval_iou, run_gate


def test_boundary_detector_finds_random_embedded_event():
    world = ContinuousWorld(seed=3, noise=0.01)
    rng = np.random.default_rng(44)

    stream = world.continuous_stream(
        actor_id=1,
        patient_id=4,
        perspective_id=0,
        object_id=2,
        total_length=64,
        start=23,
        event_length=17,
        rng=rng,
    )

    start, end = discover_boundaries(stream)

    assert interval_iou(start, end, 23, 40) > 0.95


def test_boundary_phase_key_survives_event_translation():
    world = ContinuousWorld(seed=4, noise=0.0)
    rng = np.random.default_rng(55)
    memory = BoundaryPhaseMemory(world)

    first = world.continuous_stream(
        2, 6, 0, 3, total_length=64, start=8, event_length=17, rng=rng
    )
    second = world.continuous_stream(
        2, 6, 0, 3, total_length=64, start=35, event_length=17, rng=rng
    )

    assert float(memory.key(first) @ memory.key(second)) > 0.999


def test_boundary_phase_key_survives_time_stretch():
    world = ContinuousWorld(seed=5, noise=0.0)
    rng = np.random.default_rng(66)
    memory = BoundaryPhaseMemory(world)

    short = world.continuous_stream(
        2, 6, 0, 3, total_length=64, start=10, event_length=9, rng=rng
    )
    long = world.continuous_stream(
        2, 6, 0, 3, total_length=64, start=24, event_length=25, rng=rng
    )

    assert float(memory.key(short) @ memory.key(long)) > 0.999


def test_attackers_do_not_track_random_boundaries():
    world = ContinuousWorld(seed=6, noise=0.0)
    rng = np.random.default_rng(77)

    stream = world.continuous_stream(
        1, 5, 0, 4, total_length=64, start=35, event_length=19, rng=rng
    )

    whole = WholeWindowMemory(world)
    fixed = FixedWindowMemory(world)

    _, whole_start, whole_end = whole.segment(stream)
    _, fixed_start, fixed_end = fixed.segment(stream)

    assert interval_iou(whole_start, whole_end, 35, 54) < 0.40
    assert interval_iou(fixed_start, fixed_end, 35, 54) == 0.0


def test_gate4_receipt():
    receipt = run_gate(seeds=16)
    assert_gate(receipt)

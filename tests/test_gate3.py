import numpy as np

from gate3_experiment import assert_gate, run_gate
from stream_gate import (
    FixedClockParser,
    PhaseParser,
    StreamWorld,
    SymmetricPairMemory,
)


def test_reversing_the_same_stream_swaps_actor_and_patient():
    world = StreamWorld(seed=3, noise=0.0)
    rng = np.random.default_rng(7)

    stream = world.event_stream(
        actor_id=1,
        patient_id=4,
        perspective_id=0,
        object_id=2,
        length=17,
        rng=rng,
    )

    parser = PhaseParser()
    forward = parser(stream)
    reverse = parser(stream[::-1].copy())

    assert float(forward.actor @ world.agents[1]) > 0.99
    assert float(forward.patient @ world.agents[4]) > 0.99
    assert float(reverse.actor @ world.agents[4]) > 0.99
    assert float(reverse.patient @ world.agents[1]) > 0.99


def test_phase_parser_survives_time_stretch():
    world = StreamWorld(seed=4, noise=0.0)
    rng = np.random.default_rng(8)
    parser = PhaseParser()

    for length in (9, 17, 25, 41):
        stream = world.event_stream(
            actor_id=2,
            patient_id=6,
            perspective_id=0,
            object_id=3,
            length=length,
            rng=rng,
        )
        event = parser(stream)

        assert float(event.actor @ world.agents[2]) > 0.99
        assert float(event.patient @ world.agents[6]) > 0.99


def test_fixed_clock_parser_breaks_when_event_is_stretched():
    world = StreamWorld(seed=5, noise=0.0)
    rng = np.random.default_rng(9)

    stream = world.event_stream(
        actor_id=2,
        patient_id=6,
        perspective_id=0,
        object_id=3,
        length=25,
        rng=rng,
    )

    event = FixedClockParser(train_length=9)(stream)
    mean_role_cosine = 0.5 * (
        float(event.actor @ world.agents[2])
        + float(event.patient @ world.agents[6])
    )

    assert mean_role_cosine < 0.70


def test_symmetric_pair_key_throws_away_direction():
    world = StreamWorld(seed=6, noise=0.0)
    rng = np.random.default_rng(10)
    memory = SymmetricPairMemory(world)

    stream = world.event_stream(
        actor_id=1,
        patient_id=5,
        perspective_id=0,
        object_id=4,
        length=17,
        rng=rng,
    )

    forward_key = memory.key(stream)
    reverse_key = memory.key(stream[::-1].copy())

    assert float(forward_key @ reverse_key) > 0.999


def test_gate3_receipt():
    receipt = run_gate(seeds=16)
    assert_gate(receipt)

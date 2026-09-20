from gate8_experiment import assert_gate, run_gate
from temporal_allocation_gate import (
    COARSE_RESOLUTION,
    FINE_RESOLUTION,
    MICRO_PAIRS,
    UNIFORM_RESOLUTION,
    TimedEvent,
    decode_order,
)


def event(order: int, pair: tuple[int, int]) -> TimedEvent:
    return TimedEvent(
        order=order,
        micro_pair=pair,
        consequence=1.0,
        amplitude=1.0,
        relevant=True,
    )


def test_fine_resolution_recovers_every_micro_order():
    for pair in MICRO_PAIRS:
        assert decode_order(
            event(+1, pair),
            FINE_RESOLUTION,
        ) == +1
        assert decode_order(
            event(-1, pair),
            FINE_RESOLUTION,
        ) == -1


def test_uniform_and_coarse_resolution_collapse_micro_order():
    for pair in MICRO_PAIRS:
        for resolution in (
            UNIFORM_RESOLUTION,
            COARSE_RESOLUTION,
        ):
            assert decode_order(
                event(+1, pair),
                resolution,
            ) == 0
            assert decode_order(
                event(-1, pair),
                resolution,
            ) == 0


def test_gate8_receipt():
    receipt = run_gate(seeds=16)
    assert_gate(receipt)

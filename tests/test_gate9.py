from dynamic_temporal_gate import (
    COARSE_RESOLUTION,
    FINE_RESOLUTION,
    MICRO_PAIRS,
    UNIFORM_RESOLUTION,
    EpochEvent,
    decode_order,
)
from gate9_experiment import assert_gate, run_gate


def event(
    order: int,
    pair: tuple[int, int],
    warp: int,
) -> EpochEvent:
    return EpochEvent(
        order=order,
        micro_pair=pair,
        warp=warp,
        consequence=1.0,
        amplitude=1.0,
        relevant=True,
    )


def test_normalized_fine_timing_survives_time_warp():
    for warp in (1, 2, 3):
        for pair in MICRO_PAIRS:
            assert decode_order(
                event(+1, pair, warp),
                FINE_RESOLUTION,
            ) == +1
            assert decode_order(
                event(-1, pair, warp),
                FINE_RESOLUTION,
            ) == -1


def test_coarse_and_uniform_timing_still_collapse_under_warp():
    for warp in (1, 2, 3):
        for pair in MICRO_PAIRS:
            for resolution in (
                COARSE_RESOLUTION,
                UNIFORM_RESOLUTION,
            ):
                assert decode_order(
                    event(+1, pair, warp),
                    resolution,
                ) == 0


def test_gate9_receipt():
    receipt = run_gate(seeds=16)
    assert_gate(receipt)

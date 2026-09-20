import numpy as np

from experiment import assert_gate, order_sensitivity, run_gate
from self_other_time import (
    AnchoredResidueMachine,
    LeakyOverwriteMachine,
    SnapshotResetMachine,
    TemporalObjectWorld,
)


def test_anchored_detour_returns_control_origin_but_keeps_write():
    world = TemporalObjectWorld(seed=3)
    machine = AnchoredResidueMachine(world)

    machine.detour(agent_id=2, object_id=4, tau_bin=1, value=1)

    assert machine.anchor_fidelity > 0.999999
    assert np.allclose(machine.active, machine.anchor)
    assert machine.residue_norm > 0.5
    assert np.linalg.norm(machine.memory) > 0.5


def test_snapshot_attacker_returns_exactly_unchanged():
    world = TemporalObjectWorld(seed=3)
    machine = SnapshotResetMachine(world)

    machine.detour(agent_id=2, object_id=4, tau_bin=1, value=1)

    assert machine.anchor_fidelity > 0.999999
    assert machine.residue_norm == 0.0
    assert np.allclose(machine.memory, 0.0)


def test_overwrite_attacker_eventually_loses_anchor():
    world = TemporalObjectWorld(seed=7)
    machine = LeakyOverwriteMachine(world, identity_leak=0.08)

    for i in range(80):
        agent_id = 1 + (i % (world.n_agents - 1))
        machine.detour(
            agent_id=agent_id,
            object_id=i % world.n_objects,
            tau_bin=i % world.n_tau,
            value=1 if i % 2 == 0 else -1,
        )

    assert machine.residue_norm > 0.5
    assert machine.anchor_fidelity < 0.5


def test_residue_is_temporally_order_sensitive():
    anchored = order_sensitivity(AnchoredResidueMachine, seed=11)
    reset = order_sensitivity(SnapshotResetMachine, seed=11)

    assert anchored > 0.01
    assert reset == 0.0


def test_full_gate_receipt():
    receipt = run_gate(seeds=32)
    assert_gate(receipt)

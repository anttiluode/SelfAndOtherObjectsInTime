# SelfAndOtherObjectsInTime

A small executable line about **self/other reference frames, object relations, temporal address, and what a perspective detour writes into a continuing process**.

This is not a theory of consciousness and not a literal brain model. Each gate isolates one computational claim and attacks it with matched controls.

## Gate 1 — return changed, but stable

A perspective detour should neither vanish nor rewrite the control origin.

| mechanism | anchor | residue | recent recall | order sensitivity | gate |
|---|---:|---:|---:|---:|---|
| snapshot reset | 1.000 | 0.000 | 0.500 | 0.000 | fail |
| anchored residue | 1.000 | 5.971 | 0.892 | 0.067 | **pass** |
| leaky overwrite | 0.035 | 5.971 | 0.892 | 0.067 | fail |

The surviving signature is:

~~~
SELF -> object -> OTHER -> object-from-OTHER -> SELF'
~~~

SELF' has the same control origin, but a different resident history.

## Gate 2 — identity without an agent ID

Gate 2 removes integer agent IDs from memory. Continuous identities are rebound into actor, patient and perspective roles.

| mechanism | role swap | SELF/OTHER | OTHER/OTHER | anchor | gate |
|---|---:|---:|---:|---:|---|
| **distributed binder** | **0.903** | **0.795** | **0.803** | **1.000** | **pass** |
| role-blind bag | 0.504 | 0.809 | 0.846 | 1.000 | fail |
| viewless | 0.931 | 0.505 | 0.501 | 1.000 | fail |
| SELF/OTHER binary | 0.778 | 0.748 | 0.496 | 1.000 | fail |
| mutable anchor | 0.903 | 0.795 | 0.803 | 0.740 | fail |

Which OTHER is active matters; a binary SELF/OTHER tag is insufficient in this toy.

## Gate 3 — time creates the role

Gate 3 removes the clean actor/patient/view/object tuple from the memory API. It receives one vector-valued temporal stream:

~~~
perspective : persistent / DC mode
object      : oscillatory mode
actor       : early event pulse
patient     : late event pulse
~~~

The actor and patient are the identities occupying different positions in event time.

| mechanism | 9 samples | 17 samples | 25 samples | decode cosine | gate |
|---|---:|---:|---:|---:|---|
| **phase relational** | **0.936** | **0.935** | **0.935** | **0.991** | **pass** |
| symmetric pair | 0.494 | 0.502 | 0.499 | 0.991 | fail |
| fixed clock | 0.936 | 0.578 | 0.225 | 0.361 | fail |

Reversing the same event stream swaps actor and patient. Normalized event phase survives time stretch; an absolute training clock does not.

This is the executable form of the core claim:

> **address says what is present; time gives the address relational meaning.**

## Gate 4 — discover the event before timing it

Gate 3 still received a pre-cut event. Gate 4 puts the relational event inside a longer ongoing stream.

Outside the event, the process contains quiet resident background plus slow drift. Event onset and duration are hidden from the mechanism. Test events appear at random positions and last from 9 to 25 samples.

The successful path is now:

~~~
ongoing process
    -> detect change boundaries
    -> establish event-relative phase
    -> recover actor / patient / view / object
    -> relational address
~~~

Training deliberately places every event at the same easy location, start 16 with length 9, so a fixed-window attacker gets the strongest possible training advantage. Testing then randomizes both onset and duration.

Across 64 deterministic worlds:

| mechanism | relation accuracy | boundary IoU | boundary MAE | gate |
|---|---:|---:|---:|---|
| **boundary + phase** | **0.897** | **0.997** | **0.054** | **pass** |
| whole stream = event | 0.504 | 0.265 | 23.526 | fail |
| fixed training window | 0.552 | 0.189 | 13.651 | fail |

So event-relative time is not useful until the system has established **which event owns that time coordinate**.

That distinction matters for the motivating object train. A continuing process can be interrupted by a bounded event, reorganize around it, then return to the ongoing process carrying the event's residue. The event boundary is what allows "early" and "late" to mean early and late **inside this event**, rather than arbitrary global clock positions.

### Important limitation

The boundary detector is hand-designed. It uses robust frame-to-frame change energy: event boundaries and within-event dynamics are much faster than the quiet background.

So Gate 4 removes oracle event cuts, but it does not yet discover what *kind* of change deserves to become an event. That is the next scientific weakness rather than something to hide.

## What remains scaffolded

The chain now has four distinct conveniences left to attack:

- the boundary criterion is designed by us;
- the temporal modes are designed by us;
- there is only one event at a time;
- relevance/value does not yet decide temporal resolution.

The next clean experiments are therefore overlapping/interrupted event trains and adaptive time allocation. Those are much closer to the mouse -> remembered person -> mouse -> self trajectory that motivated the repo: one event can recruit another perspective/event while the first process is still resident.

## Run

~~~bash
python -m pip install -r requirements.txt
python experiment.py --assert-gate --seeds 64
python gate2_experiment.py --assert-gate --seeds 64
python gate3_experiment.py --assert-gate --seeds 64
python gate4_experiment.py --assert-gate --seeds 64
pytest -q
~~~

Receipts are committed under results/gate1.json through results/gate4.json.

## Scientific inspirations, not equivalences

The direction was motivated by empirical work on timing, emotion/time interaction, and distributed phase coordination:

- Rolando et al., *Current Biology* (2024), **Distinct neural adaptations to time demand in the striatum and the hippocampus**, DOI: 10.1016/j.cub.2023.11.066.
- Doyère & Droit-Volet, *Cerebral Cortex* (2025), **When emotion and time meet from human and rodent perspectives: a central role for the amygdala?**, DOI: 10.1093/cercor/bhae454.
- Ye et al., bioRxiv (version posted 2025), **Brain-wide topographic coordination of traveling spiral waves**, DOI: 10.1101/2023.12.07.570517.

Those papers do not demonstrate the mechanisms in this repository.

## Relation to the older repo line

- **FrequencyAddressedState-dependentOperatorComposition** — address plus resident state; temporal phase now contributes relational role.
- **FusionMachine** — computations remain resident while another process is selected.
- **Sihti / SighImageFactorization** — Gates 3–4 are temporal factorization: separate superposed components by modes after finding the event that owns them.
- **the_whorl / ArtificialCortex** — suggests future phase coordinates generated by the substrate rather than a global normalized phase.
- **AnotherOddThing** — suggests actively selecting which interruption/event/perspective is worth resolving.

The target remains narrow: **a stable reference process whose object relations acquire meaning from the temporal path through bounded events**.

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

The successful path is:

~~~
ongoing process
    -> detect change boundaries
    -> establish event-relative phase
    -> recover actor / patient / view / object
    -> relational address
~~~

Across 64 deterministic worlds:

| mechanism | relation accuracy | boundary IoU | boundary MAE | gate |
|---|---:|---:|---:|---|
| **boundary + phase** | **0.897** | **0.997** | **0.054** | **pass** |
| whole stream = event | 0.504 | 0.265 | 23.526 | fail |
| fixed training window | 0.552 | 0.189 | 13.651 | fail |

So event-relative time is not useful until the system has established **which event owns that time coordinate**.

The boundary detector is still hand-designed. Gate 4 removes oracle event cuts, not the criterion for what deserves to become an event.

## Gate 5 — interruption requires an event stack

Gate 4 still allowed only one active event.

Gate 5 gives the parent event an interruption:

~~~
PARENT early
    -> CHILD opens
        -> child event runs on its own local phase
        -> child closes
    -> PARENT resumes
PARENT closes
~~~

The boundary markers stand in for already-detected open/close transitions from a Gate-4-like detector. Gate 5 isolates what the mechanism must do **after** those transitions are known.

The successful mechanism uses stack semantics:

1. pause the parent's local event phase;
2. push the child;
3. run the child on its own normalized phase;
4. close the child;
5. write the child's relational result into the parent;
6. resume the parent's old phase rather than restarting it.

The matched challenge is deliberately hard on shortcuts. Each pair contains the **same parent event**, two different children, and opposite labels. Therefore preserving the parent alone cannot solve it. The child must change the resumed parent address.

Training uses parent length 17, child length 7, insertion point 8. Testing randomizes parent duration, child duration from 7 to 31, and interruption position.

Across 64 deterministic worlds:

| mechanism | all test episodes | long child (>=20) | gate |
|---|---:|---:|---|
| **nested stack + writeback** | **0.994** | **0.993** | **pass** |
| flat global timeline | 0.768 | 0.754 | fail |
| reset parent after child | 0.500 | 0.510 | fail |
| preserve parent, no child writeback | 0.504 | 0.518 | fail |

The failures separate three different requirements.

**Flat timeline** still sees all the content, so it performs above chance. But increasing child duration pushes the parent's late role around a single global clock and degrades the relation.

**Reset on resume** throws away the pre-interruption parent. It can return to parent-like activity but not to the unfinished parent event.

**No writeback** preserves parent continuity perfectly well, but paired trials become indistinguishable because the child never changes the parent state.

So the useful signature is now:

~~~
PARENT
  -> CHILD
  -> CHILD'
  -> PARENT'
~~~

with two simultaneous facts:

~~~
PARENT' is still the same unfinished parent event
AND
PARENT' is changed by what happened in the child
~~~

That is the recursive version of Gate 1.

It also makes the phrase **"time has an owner"** more precise. The parent and child can have different local clocks at the same point in the global stream. The child's duration should not advance the parent's event phase merely because wall-clock time passed.

## Where this leaves the motivating sequence

A trajectory such as:

~~~
ongoing self process
    -> mouse/place event
        -> remembered-person / other-perspective event
        -> mouse/place event'
    -> ongoing self process'
~~~

can now be represented without forcing everything onto one timeline or pretending the interruption vanished.

This does **not** imply that human cognition literally implements a pushdown stack. The gate establishes only that this toy needs stack-like state semantics to preserve nested local time and child-to-parent residue under the matched attacks.

## What remains scaffolded

The strongest remaining cheats are now clearer:

- Gate 4's criterion for opening/closing an event is still designed by us;
- Gate 3's temporal modes are still designed by us;
- Gate 5 receives explicit nesting transitions after boundary detection;
- there is only one level of child nesting in the benchmark;
- relevance/value does not yet determine whether an interruption deserves its own event or how much temporal resolution it receives.

The next useful attack is therefore not "add more memory." It is **event admission**:

> two changes happen while a parent event is active; which one deserves to open a child event?

That is where the line can reconnect to AnotherOddThing / Sihti-style active selection and to relevance-dependent time allocation without smuggling the answer into the event marker.

## Run

~~~bash
python -m pip install -r requirements.txt
python experiment.py --assert-gate --seeds 64
python gate2_experiment.py --assert-gate --seeds 64
python gate3_experiment.py --assert-gate --seeds 64
python gate4_experiment.py --assert-gate --seeds 64
python gate5_experiment.py --assert-gate --seeds 64
pytest -q
~~~

Receipts are committed under results/gate1.json through results/gate5.json.

## Scientific inspirations, not equivalences

The direction was motivated by empirical work on timing, emotion/time interaction, and distributed phase coordination:

- Rolando et al., *Current Biology* (2024), **Distinct neural adaptations to time demand in the striatum and the hippocampus**, DOI: 10.1016/j.cub.2023.11.066.
- Doyère & Droit-Volet, *Cerebral Cortex* (2025), **When emotion and time meet from human and rodent perspectives: a central role for the amygdala?**, DOI: 10.1093/cercor/bhae454.
- Ye et al., bioRxiv (version posted 2025), **Brain-wide topographic coordination of traveling spiral waves**, DOI: 10.1101/2023.12.07.570517.

Those papers do not demonstrate the mechanisms in this repository.

## Relation to the older repo line

- **FrequencyAddressedState-dependentOperatorComposition** — address plus resident state; Gate 5 adds nested local clocks and writeback.
- **FusionMachine** — the parent computation remains resident while the child computation is selected and executed.
- **Sihti / SighImageFactorization** — temporal components stay separable rather than being flattened into one mixed history.
- **AnotherOddThing** — the next gate can actively decide which interruption deserves a new event context.
- **ReadWrite** — Gate 5's child close is explicitly a write into the resumed parent state.
- **the_whorl / ArtificialCortex** — future work can replace global event phase with a locally generated substrate phase.

The target remains narrow: **a stable reference process whose relations live in bounded, nestable local times and whose completed detours can change the process that resumes**.

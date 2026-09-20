# SelfAndOtherObjectsInTime

A small executable line about **self/other reference frames, object relations, temporal address, and what a perspective detour writes into a continuing process**.

This is not a theory of consciousness and not a literal brain model. Each gate isolates one computational claim and attacks it with matched controls.

## Gate 1 — return changed, but stable

Gate 1 asks:

> After a system runs another perspective and returns to itself, should it be exactly the same state as before?

The tested signature is:

~~~
return to A
+ A remains the control origin
+ A is not unchanged
~~~

Three mechanisms were compared:

- **snapshot reset**: stable SELF, but the detour is erased;
- **leaky overwrite**: the detour remains, but simulated perspectives rewrite SELF;
- **anchored residue**: the active perspective can move, the control anchor stays fixed, and the detour writes resident state.

Across 64 deterministic worlds:

| mechanism | anchor | residue | recent recall | order sensitivity | gate |
|---|---:|---:|---:|---:|---|
| snapshot reset | 1.000 | 0.000 | 0.500 | 0.000 | fail |
| anchored residue | 1.000 | 5.971 | 0.892 | 0.067 | **pass** |
| leaky overwrite | 0.035 | 5.971 | 0.892 | 0.067 | fail |

So Gate 1 makes the prime executable:

~~~
SELF -> object -> OTHER -> object-from-OTHER -> SELF'
~~~

SELF' has the same control anchor, but a different resident history.

## Gate 2 — identity without an agent ID

Gate 1 still handed the machine an explicit agent slot. Gate 2 removes that convenience.

The machine now receives only noisy continuous vectors for actor, patient, current perspective, object, and event-relative time. Hidden integer IDs exist only inside the evaluator so it can construct matched pairs. They are never passed to the mechanism.

The same distributed identity can occupy actor, patient, or perspective. Fixed signed role masks move that identity into a role-specific address:

~~~
key =
    bind(ACTOR, actor_vector)
  + bind(PATIENT, patient_vector)
  + bind(VIEW, perspective_vector)
  + selfness(perspective, anchor)
  + bind(OBJECT, object_vector)
  + bind(TIME, tau_vector)
~~~

Selfness is not a SELF token. It is a graded similarity between the current perspective cue and the protected control anchor.

### Matched attacks

Every challenge pair has the same object/time content and opposite hidden labels, while changing only one relational fact.

1. **role swap**: A acts-on B versus B acts-on A;
2. **self/other perspective**: same event viewed from SELF versus an OTHER;
3. **other/other perspective**: same event viewed from OTHER-1 versus OTHER-2.

Fresh noisy vectors are used at test time, so this is not exact-vector lookup.

Across 64 deterministic worlds:

| mechanism | role swap | SELF/OTHER | OTHER/OTHER | anchor | gate |
|---|---:|---:|---:|---:|---|
| **distributed binder** | **0.903** | **0.795** | **0.803** | **1.000** | **pass** |
| role-blind bag | 0.504 | 0.809 | 0.846 | 1.000 | fail |
| viewless | 0.931 | 0.505 | 0.501 | 1.000 | fail |
| SELF/OTHER binary | 0.778 | 0.748 | 0.496 | 1.000 | fail |
| mutable anchor | 0.903 | 0.795 | 0.803 | 0.740 | fail |

Each attacker fails for a different reason: a bag of participants cannot represent direction; an event without perspective cannot represent viewpoint; a binary SELF/OTHER flag cannot represent **which other** is currently instantiated; and relational competence is not enough if every visited viewpoint rewrites the control anchor.

So Gate 2 needs both:

~~~
stable control origin
AND
nontrivial perspective-specific writes
~~~

### What Gate 2 does not establish

Actor, patient, perspective, object and event-relative time are still supplied as structured input channels, and the role-binding masks are fixed. Gate 2 removes agent IDs from memory; it does not yet discover the relational roles themselves.

That is the next weakness.

## Time is already part of the address

Both gates still receive event-relative time, tau. The same participants and object at different positions inside an event need not mean the same thing.

But tau is currently handed in. A later gate should replace it with internally generated temporal structure: event boundaries, elapsed-time populations, oscillatory phase, and eventually propagating phase fields.

## Run

~~~bash
python -m pip install -r requirements.txt
python experiment.py --assert-gate --seeds 64
python gate2_experiment.py --assert-gate --seeds 64
pytest -q
~~~

Deterministic receipts are in results/gate1.json and results/gate2.json.

## Interactive microscope

The root index.html is the GitHub Pages microscope. Gate 1 animates the perspective detour. Gate 2 shows why role, perspective identity, and anchor protection are independent requirements.

## Scientific inspirations, not equivalences

The direction was motivated by empirical work on timing, emotion/time interaction, and distributed phase coordination:

- Rolando et al., *Current Biology* (2024), **Distinct neural adaptations to time demand in the striatum and the hippocampus**, DOI: 10.1016/j.cub.2023.11.066.
- Doyère & Droit-Volet, *Cerebral Cortex* (2025), **When emotion and time meet from human and rodent perspectives: a central role for the amygdala?**, DOI: 10.1093/cercor/bhae454.
- Ye et al., bioRxiv (version posted 2025), **Brain-wide topographic coordination of traveling spiral waves**, DOI: 10.1101/2023.12.07.570517.

Those papers do not demonstrate the mechanisms in this repository. They motivate separating event boundaries, elapsed time, relevance/value, and distributed coordination.

## Relation to the older repo line

- **FrequencyAddressedState-dependentOperatorComposition** — address plus resident state; this repo adds perspective and event-relative time to the address.
- **FusionMachine** — multiple computations can remain resident; here a visited viewpoint becomes one of the residues.
- **Sihti / SighImageFactorization** — preserve residue rather than pretending a transformation erased everything.
- **the_whorl / ArtificialCortex** — geometry-generated phase fields suggest a route where temporal address is generated by the substrate rather than handed in.
- **AnotherOddThing** — later gates can choose which perspective/probe to instantiate rather than passively receiving a fixed detour.

## Next attacks

Gate 2 removes explicit agent IDs, but there is still scaffolding:

- remove the supplied actor/patient/view role channels and infer relations from a temporal stream;
- test unseen identities and unseen object combinations;
- reverse event order while preserving the same object multiset;
- time-stretch an event and demand relational rather than absolute timing;
- let value/relevance alter temporal resolution without directly revealing the answer;
- replace supplied tau with an internally generated phase/elapsed-time field;
- make useful perspective residue compete with contamination of the control anchor.

The target remains narrow: **a mechanism that keeps a stable reference origin while the paths it computes become part of what it can do next**.

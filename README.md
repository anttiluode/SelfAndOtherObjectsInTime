# SelfAndOtherObjectsInTime

A small executable line about **self/other reference frames, object relations, temporal address, and what a perspective detour writes into a continuing process**.

This is not a theory of consciousness and not a literal brain model. Each gate isolates one computational claim and attacks it with matched controls.

## Gate 1 — return changed, but stable

The first gate asks whether a perspective detour should be perfectly reversible.

Across 64 deterministic worlds:

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

Gate 2 removes integer agent IDs from the memory mechanism. It receives only noisy continuous vectors for actor, patient, perspective, object and event-relative time.

The same distributed identity can occupy different roles.

Across 64 worlds:

| mechanism | role swap | SELF/OTHER | OTHER/OTHER | anchor | gate |
|---|---:|---:|---:|---:|---|
| **distributed binder** | **0.903** | **0.795** | **0.803** | **1.000** | **pass** |
| role-blind bag | 0.504 | 0.809 | 0.846 | 1.000 | fail |
| viewless | 0.931 | 0.505 | 0.501 | 1.000 | fail |
| SELF/OTHER binary | 0.778 | 0.748 | 0.496 | 1.000 | fail |
| mutable anchor | 0.903 | 0.795 | 0.803 | 0.740 | fail |

A binary SELF/OTHER flag is therefore not enough in this toy: which other perspective is active matters too.

But Gate 2 still supplied clean role channels.

## Gate 3 — time creates the role

Gate 3 removes the actor, patient, perspective, object and tau channels from the memory API.

The mechanism now receives **one vector-valued temporal stream**.

Four components ride different temporal modes inside that stream:

~~~
perspective : persistent / DC mode
object      : oscillatory mode
actor       : early event pulse
patient     : late event pulse
~~~

The phase parser decomposes the event with a small temporal basis. The actor and patient are therefore not different ID slots. They are the identities occupying different positions in event time.

That makes reversal mechanical:

~~~
A -> B
reverse time
B -> A
~~~

The stream contains the same identities and object either way. Temporal order supplies the relational meaning.

### Time-stretch attack

Training events are 9 samples long. The same event is then replayed at 17 and 25 samples.

The successful mechanism uses **normalized event phase** rather than a fixed absolute clock.

Across 64 deterministic worlds:

| mechanism | 9 samples | 17 samples | 25 samples | decomposition cosine | gate |
|---|---:|---:|---:|---:|---|
| **phase relational** | **0.936** | **0.935** | **0.935** | **0.991** | **pass** |
| symmetric pair | 0.494 | 0.502 | 0.499 | 0.991 | fail |
| fixed clock | 0.936 | 0.578 | 0.225 | 0.361 | fail |

The two negative controls separate two ideas.

**Symmetric pair** decomposes the stream almost perfectly, but then adds actor and patient together. It knows who was present and still loses the relation. Correct decomposition alone is not enough; direction must survive into the address.

**Fixed clock** works at the exact training duration, then collapses when the same event is stretched. Absolute sample number is not the useful invariant here. Relative position inside the bounded event is.

The reverse-stream diagnostic is even cleaner: the phase parser decodes a reversed 25-sample stream with mean cosine **0.991** to the swapped actor/patient identities.

So Gate 3 turns the phrase from the motivating discussion into an executable claim:

> **address says what is present; time gives the address relational meaning.**

## What is still handed in

Gate 3 is stronger than Gate 2, but the temporal basis itself is still designed by us. Event start and end are known, which is what allows normalized phase.

The next attack should therefore remove one of those conveniences rather than adding more decorative complexity:

- infer event boundaries instead of receiving a pre-cut event;
- learn or self-organize the temporal modes instead of supplying their shapes;
- test overlapping/nested events where one object train interrupts another;
- make relevance/value change the temporal resolution allocated to an event;
- eventually replace a globally supplied phase coordinate with a locally generated or propagating field.

That is where the striatum / hippocampal-boundary / oscillatory-wave story can begin to touch the toy without being baked into it.

## Run

~~~bash
python -m pip install -r requirements.txt
python experiment.py --assert-gate --seeds 64
python gate2_experiment.py --assert-gate --seeds 64
python gate3_experiment.py --assert-gate --seeds 64
pytest -q
~~~

Receipts:

- results/gate1.json
- results/gate2.json
- results/gate3.json

## Scientific inspirations, not equivalences

The direction was motivated by empirical work on timing, emotion/time interaction, and distributed phase coordination:

- Rolando et al., *Current Biology* (2024), **Distinct neural adaptations to time demand in the striatum and the hippocampus**, DOI: 10.1016/j.cub.2023.11.066.
- Doyère & Droit-Volet, *Cerebral Cortex* (2025), **When emotion and time meet from human and rodent perspectives: a central role for the amygdala?**, DOI: 10.1093/cercor/bhae454.
- Ye et al., bioRxiv (version posted 2025), **Brain-wide topographic coordination of traveling spiral waves**, DOI: 10.1101/2023.12.07.570517.

Those papers do not demonstrate the mechanisms in this repository.

## Relation to the older repo line

- **FrequencyAddressedState-dependentOperatorComposition** — address plus resident state; Gate 3 now lets temporal mode choose relational role.
- **FusionMachine** — multiple computations can remain resident; a visited viewpoint can persist as residue.
- **Sihti / SighImageFactorization** — Gate 3 is explicitly a tiny temporal factorization: separate superposed components by mode.
- **the_whorl / ArtificialCortex** — suggests the future phase coordinate could be generated by the substrate rather than supplied globally.
- **AnotherOddThing** — later gates can choose which event/perspective to interrogate.

The target remains narrow: **a stable reference process whose object relations acquire meaning from the temporal path through them**.

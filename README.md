# SelfAndOtherObjectsInTime

A small executable line about **self/other reference frames, object relations, temporal address, nested events, and what a detour writes into a continuing process**.

This is not a theory of consciousness and not a literal brain model. Each gate isolates one computational claim and attacks it with matched controls.

## Gate 1 — return changed, but stable

A perspective detour should neither vanish nor rewrite the control origin.

| mechanism | anchor | residue | recent recall | order sensitivity | gate |
|---|---:|---:|---:|---:|---|
| snapshot reset | 1.000 | 0.000 | 0.500 | 0.000 | fail |
| anchored residue | 1.000 | 5.971 | 0.892 | 0.067 | **pass** |
| leaky overwrite | 0.035 | 5.971 | 0.892 | 0.067 | fail |

The surviving signature is SELF -> object -> OTHER -> object-from-OTHER -> SELF'. SELF' has the same control origin, but a different resident history.

## Gate 2 — identity without an agent ID

Continuous identities can be rebound into actor, patient and perspective roles.

| mechanism | role swap | SELF/OTHER | OTHER/OTHER | anchor | gate |
|---|---:|---:|---:|---:|---|
| **distributed binder** | **0.903** | **0.795** | **0.803** | **1.000** | **pass** |
| role-blind bag | 0.504 | 0.809 | 0.846 | 1.000 | fail |
| viewless | 0.931 | 0.505 | 0.501 | 1.000 | fail |
| SELF/OTHER binary | 0.778 | 0.748 | 0.496 | 1.000 | fail |
| mutable anchor | 0.903 | 0.795 | 0.803 | 0.740 | fail |

Which OTHER is active matters; a binary SELF/OTHER tag is insufficient in this toy.

## Gate 3 — time creates the role

One vector-valued event stream carries perspective, object, actor and patient in different temporal modes.

| mechanism | 9 samples | 17 samples | 25 samples | decode cosine | gate |
|---|---:|---:|---:|---:|---|
| **phase relational** | **0.936** | **0.935** | **0.935** | **0.991** | **pass** |
| symmetric pair | 0.494 | 0.502 | 0.499 | 0.991 | fail |
| fixed clock | 0.936 | 0.578 | 0.225 | 0.361 | fail |

Reversing the same stream swaps actor and patient. Event-relative phase survives time stretch; an absolute training clock does not.

> **address says what is present; time gives the address relational meaning.**

## Gate 4 — discover the event before timing it

The event is embedded inside an ongoing process. Onset and duration are hidden.

| mechanism | relation accuracy | boundary IoU | boundary MAE | gate |
|---|---:|---:|---:|---|
| **boundary + phase** | **0.897** | **0.997** | **0.054** | **pass** |
| whole stream = event | 0.504 | 0.265 | 23.526 | fail |
| fixed training window | 0.552 | 0.189 | 13.651 | fail |

Event-relative time becomes meaningful only after the process establishes **which event owns the clock**.

## Gate 5 — interruption requires nested local time

A child event can interrupt an unfinished parent.

| mechanism | all episodes | long child >=20 | gate |
|---|---:|---:|---|
| **nested stack + writeback** | **0.994** | **0.993** | **pass** |
| flat global timeline | 0.768 | 0.754 | fail |
| reset parent after child | 0.500 | 0.510 | fail |
| preserve parent, no writeback | 0.504 | 0.518 | fail |

The parent and child can own different local clocks at the same point in global time. Closing the child writes into the unfinished parent instead of either resetting it or leaving it unchanged.

## Gate 6 — magnitude is not event-worthiness

Gate 5 still assumed that an interruption had already earned a child context.

Gate 6 gives the active parent **two candidate interruptions but only one child-event slot**.

One interruption is large locally but almost null downstream. The other is much smaller locally but strongly changes a known downstream readout after transport.

The successful admission score is:

    s(c) = |m^T J delta_c|

not local magnitude:

    s(c) = ||delta_c||

Across 64 deterministic worlds, the distractor is on average **8.245x larger in raw amplitude**, while the relevant candidate is **24.914x larger in downstream consequence**.

| mechanism | prediction | admission precision | gate |
|---|---:|---:|---|
| **downstream consequence** | **0.987** | **1.000** | **pass** |
| raw magnitude | 0.492 | 0.000 | fail |
| admit every disturbance | 0.502 | 0.500 | fail |
| ignore interruptions | 0.497 | 0.000 | fail |

The relevant child relation is stable between training and testing, while the large irrelevant distractor is replaced by a fresh unseen relation. If the system writes every disturbance into the parent, the parent is contaminated by something that never mattered to its future.

The admission signal cannot leak the answer: matched episodes share the same parent and the same consequence-score magnitude, but use different relevant child contents with opposite labels. Mean matched score gap is effectively zero (3.37e-17).

So Gate 6 sharpens the idea of salience:

> **not "what changed most?" but "what is predicted to change the continuing process downstream?"**

### Important limitation

Gate 6 is not yet active causal discovery. The transport operator J and downstream readout m are known to the mechanism.

That is deliberate scaffolding.

The next attack is: can the system learn which interruption is consequential from interventions and observed downstream changes, instead of receiving the causal transport?

That is where AnotherOddThing and ReadWrite connect directly to this repo.


## Gate 7 — discover consequence by choosing experiments

Gate 6 knew the causal consequence map. Gate 7 hides it.

The world contains one of eight possible downstream-sensitivity hypotheses. The mechanism gets a strict budget of three noisy interventions before it must decide which future interruption deserves a child event.

Active probing chooses the next intervention by expected information gain. Random probing gets the exact same three-probe budget.

After probing, both use the same admission rule:

    score(delta) = E_h[ |w_h^T delta| ]

where the expectation is under the current posterior over causal hypotheses.

Across 64 deterministic worlds:

| mechanism | prediction | relevant admission | hidden-map ID | posterior entropy | gate |
|---|---:|---:|---:|---:|---|
| oracle map | 0.988 | 1.000 | 1.000 | 0.000 | control |
| **active EIG** | **0.966** | **0.954** | **0.953** | **0.424** | **pass** |
| random probes | 0.690 | 0.433 | 0.531 | 1.006 | fail |
| no probes | 0.495 | 0.000 | 0.047 | 3.000 | fail |

Active and random both spend exactly three probes. Active chooses three nonredundant probe categories in every world; random averages 2.375.

This is the direct bridge to the older active-experiment line: event admission no longer asks only "what matters downstream?" It asks:

> **what experiment should I perform now so I can know what will matter downstream later?**

### Important limitation

The hypothesis family and probe likelihood model are still supplied. Gate 7 identifies which causal map is present; it does not yet learn an arbitrary transport from raw experience.

The next clean attack is therefore either continuous transport learning or relevance-dependent temporal resolution: once an event is admitted, how much timing precision should it receive?

## Run

    python -m pip install -r requirements.txt
    python experiment.py --assert-gate --seeds 64
    python gate2_experiment.py --assert-gate --seeds 64
    python gate3_experiment.py --assert-gate --seeds 64
    python gate4_experiment.py --assert-gate --seeds 64
    python gate5_experiment.py --assert-gate --seeds 64
    python gate6_experiment.py --assert-gate --seeds 64
    python gate7_experiment.py --assert-gate --seeds 64
    pytest -q

Receipts are committed under results/gate1.json through results/gate7.json.

## Scientific inspirations, not equivalences

The direction was motivated by empirical work on timing, emotion/time interaction, and distributed phase coordination:

- Rolando et al., Current Biology (2024), Distinct neural adaptations to time demand in the striatum and the hippocampus, DOI: 10.1016/j.cub.2023.11.066.
- Doyère & Droit-Volet, Cerebral Cortex (2025), When emotion and time meet from human and rodent perspectives: a central role for the amygdala?, DOI: 10.1093/cercor/bhae454.
- Ye et al., bioRxiv (version posted 2025), Brain-wide topographic coordination of traveling spiral waves, DOI: 10.1101/2023.12.07.570517.

Those papers do not demonstrate the mechanisms in this repository.

## Relation to the older repo line

- **FrequencyAddressedState-dependentOperatorComposition** — address plus resident state; later gates add event-local phase and event admission.
- **FusionMachine** — the parent computation remains resident while selected child computations run.
- **Sihti / SighImageFactorization** — keep consequential components and residues separable instead of flattening everything.
- **AnotherOddThing** — Gate 7 now directly reuses the expected-information-gain idea to choose which causal probe to run under a matched budget.
- **ReadWrite** — Gate 6 supplied the observability/write map; Gate 7 now identifies which map is present from interventions and downstream observations.
- **the_whorl / ArtificialCortex** — future work can replace globally supplied event phase with locally generated substrate phase.

The target remains narrow: **a stable reference process whose relations live in bounded local times, whose detours write back, and whose event boundaries are allocated to changes that matter downstream rather than merely changes that are large**.

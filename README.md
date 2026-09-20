# SelfAndOtherObjectsInTime

A tiny computational test of one question:

> After a system runs another agent's perspective and returns to itself, should it be exactly the same state as before?

Gate 1 says **no**. A useful self can return to a stable control origin while retaining a residue of the perspective it just computed.

This repository is deliberately not a theory of consciousness. It is a falsifiable toy about **self/other reference frames, object relations, temporal address, and state-writing perspective switches**.

## Gate 1 — changed but stable

We compare three mechanisms under the same synthetic perspective-detour stream.

1. **Snapshot reset** — enter the other perspective, then restore the complete pre-detour state. Identity is stable, but the detour leaves no autobiographical write.
2. **Leaky overwrite** — the detour writes useful residue, but the self anchor itself is allowed to drift toward every simulated agent.
3. **Anchored residue** — the active perspective may move, but the control anchor is protected while the detour writes into resident associative state.

The tested signature is not "can I impersonate B?" It is:

[
	ext{return to A} quad+quad 	ext{A remains A} quad+quad 	ext{A is not unchanged}.
]

The machine state is split into a control anchor (a_t), an active perspective, a resident vector (h_t), and associative memory (M_t). A perspective/object/time event receives a distributed key

[
k_t = operatorname{norm}(e_{	ext{agent}} + e_{	ext{object}} + 0.7e_{	au}),
]

where (	au) is **event-relative time**, not wall-clock time. A detour writes

[
M_{t+1}=lambda M_t+y_t k_tk_t^	op,qquad
h_{t+1}=lambda h_t+y_tk_t.
]

For the anchored-residue mechanism the active perspective returns to the self anchor, but (M) and (h) are not rolled back.

### Gate criteria

Across 64 deterministic worlds, a mechanism counts as **changed but stable** only if it simultaneously has:

- mean self-anchor cosine (>0.98),
- mean residue norm (>0.5),
- recent perspective-event recall (>0.75),
- nonzero order sensitivity (>0.01).

Current deterministic receipt:

| mechanism | anchor fidelity | residue norm | recent recall | order sensitivity | gate |
|---|---:|---:|---:|---:|---|
| snapshot reset | 1.000 | 0.000 | 0.500 | 0.000 | fail |
| anchored residue | 1.000 | 5.971 | 0.892 | 0.067 | **pass** |
| leaky overwrite | 0.035 | 5.971 | 0.892 | 0.067 | fail |

These numbers are a **mechanism receipt**, not evidence about human selfhood.

Run it:

```bash
python -m pip install -r requirements.txt
python experiment.py --assert-gate
pytest -q
```

## The object train

The motivating sequence can be written without any biography:

```
SELF / introspection
        ↓ interrupted by
object
        ↓ relation evokes
OTHER
        ↓ perspective is instantiated
object-from-OTHER
        ↓ return
SELF'
```

The prime matters:

[
SELF' 
eq SELF
]

in resident history, while the control origin is still the same self.

That is the distinction this first gate makes executable. Perspective-taking is not modeled as a perfectly reversible coordinate transform. It is a **state transition that can write**.

## Why "in time" is already in Gate 1

Object identity alone is insufficient. The same agent/object relation at different positions inside an event can have different meaning, so keys include an event-relative temporal address (	au).

This first implementation uses a small learned-free temporal basis only to make the distinction testable. Later gates should replace that convenience with mechanisms that generate their own temporal coordinates: phase, elapsed-time populations, boundaries, and eventually propagating fields.

## Interactive microscope

The root `index.html` is a conceptual microscope for the same three mechanisms. It animates

```
SELF → object → OTHER → object-from-OTHER → SELF'
```

and makes the two independent requirements visible: **anchor stability** and **persistent residue**.

GitHub Pages workflow was already present when the repository was opened.

## Scientific inspirations, not equivalences

The current direction was motivated by three empirical lines:

- Rolando et al., *Current Biology* (2024), **Distinct neural adaptations to time demand in the striatum and the hippocampus**, DOI: 10.1016/j.cub.2023.11.066.
- Doyère & Droit-Volet, *Cerebral Cortex* (2025), **When emotion and time meet from human and rodent perspectives: a central role for the amygdala?**, DOI: 10.1093/cercor/bhae454.
- Ye et al., bioRxiv (version posted 2025), **Brain-wide topographic coordination of traveling spiral waves**, DOI: 10.1101/2023.12.07.570517.

None of those papers demonstrates the mechanism implemented here. They motivate the separation of event boundaries, elapsed time, value/relevance, and distributed phase coordination. This repository tests a much narrower computational claim.

## Relation to the older repo line

- **FrequencyAddressedState-dependentOperatorComposition** — address plus resident state; this repo adds perspective and event-relative time to the address.
- **FusionMachine** — multiple computations can remain resident; here another agent's viewpoint can become one of those residues.
- **Sihti / SighImageFactorization** — preserve residue rather than pretending a transformation erased everything; here the "residue" is a perspective detour.
- **the_whorl / ArtificialCortex** — geometry-generated phase fields suggest a future route where temporal address is generated by the substrate rather than handed in.
- **AnotherOddThing** — later gates should actively choose which perspective/probe to instantiate rather than passively receiving a fixed detour stream.

## Next gates

Gate 1 intentionally makes the distinction before making it biological.

The next attacks are already obvious:

- remove explicit agent slots and require distributed self/other binding;
- require return-after-detour under unseen agents and unseen objects;
- reverse event order while preserving the same object multiset;
- time-stretch the same event and demand relational, not absolute, timing;
- let value/relevance alter temporal resolution without directly revealing the answer;
- replace the supplied (	au) basis with an internally generated phase/elapsed-time field;
- make perspective residue useful for prediction but dangerous if it contaminates the control anchor.

The target is **not** a little conscious machine. The target is a mechanism that can keep a stable reference origin while letting the paths it computes become part of its future state.

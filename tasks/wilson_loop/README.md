# Task: Wilson Loop

Optimize the spatial Wilson line operator to improve ground-state overlap in static quark-antiquark Wilson loop measurements.

## Background

The Wilson loop `W_{r x t}` measures the potential between a static quark and antiquark separated by distance `r`. The correlator decays as:

```
<tr W_{r x t}> = sum_n |c_n|^2 exp(-t * a * E_n(r))
```

The spatial Wilson line connecting the quark and antiquark determines how well the operator projects onto the ground state. A straight product of links (the "plain" Wilson line) has poor overlap — you can do much better.

## What You Optimize

You implement `SpatialOperator` from `interface.py`. Your operator replaces the straight spatial Wilson line with something that has better ground-state overlap `|c_0|^2`.

**Constraint**: your operator must be gauge-covariant. Under a gauge transformation, it must transform as `G(x) @ S(x) @ G^dag(x+r)`. Products and sums of gauge link paths from `x` to `x+r` automatically satisfy this.

## Getting Started

1. Look at `operators/plain.py` for the simplest baseline
2. Read the [reference paper](../../references/2602.02436.pdf) for ideas (neural network layers, plaquette insertions, smearing)
3. Create your operator in `operators/your_name.py`
4. Test: `pytest tests/ -k your_operator`
5. Benchmark: `python benchmark/run.py --operator your_name`

## Baselines

| Operator | Description | Score |
|---|---|---|
| `plain` | Straight Wilson line (product of links) | TBD |

## Hints

- Inserting plaquettes (clover-leaf field-strength terms) along the path adds chromo-electromagnetic structure
- APE/HYP smearing of spatial links reduces UV noise
- Coulomb gauge fixing makes spatial links smooth (but breaks manifest gauge invariance)
- The paper's gauge-equivariant layers (linear, bilinear, convolutional) are systematic ways to build better operators

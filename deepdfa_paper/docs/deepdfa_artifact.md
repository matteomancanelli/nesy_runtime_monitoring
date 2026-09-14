# DeepDFA artifact guide

This repository implements the forward computation used by DeepDFA in two
fixed representations of an LTLf-compiled DFA. It is an exact runtime monitor
for a known specification, not an implementation of DeepDFA automaton learning.

## Scope and semantics

- **Dense mode** stores the one-hot tensor
  `T[|Q|, 2^|AP|, |Q|]` and accepts crisp observations.
- **Factored mode** stores pairwise-disjoint guard cubes. It is exact for crisp
  observations and computes exact weighted model counts for fractional inputs
  under the independent-Bernoulli atom model.
- **Scan mode** uses the same crisp transition matrices and a Hillis--Steele
  prefix product. It changes launch depth and memory use, not semantics.
- `method="recursive"` is a diagnostic approximation. It is exact on crisp
  inputs and on fractional inputs when the relevant guard subexpressions are
  independent (including read-once guards). Use the default `method="exact"`
  for probabilistic monitoring.

The probabilistic output is the marginal probability that the random trace is
accepted. Atom values are assumed independent within a cell, and cells are
assumed independent across time. It is not a three-valued runtime verdict.
`soft_verdict` is only a convenience thresholding operation.

The artifact does **not** contain trainable transition logits, temperature
annealing, a learned acceptance vector, DFA extraction, or post-training
minimization. Those mechanisms are needed for specification learning or
adaptation, not for monitoring a specification that is already known.

## Public API

```python
from dataclasses import asdict

import torch

from src.monitors import DeepDFAMonitorFactored

monitor = DeepDFAMonitorFactored.compile("F (a & b)")

# Exact crisp monitoring; omitted atoms are false.
verdict = monitor.run([
    {"a": True},
    {"a": True, "b": True},
])

# Exact differentiable acceptance marginal. Shape is (L, |AP|).
p = torch.tensor(
    [[0.8, 0.2], [0.7, 0.9]],
    dtype=torch.float32,
    requires_grad=True,
)
score = monitor.acceptance_probability_tensor(p)
score.backward()

# Stable representation diagnostics suitable for JSON/CSV artifact reports.
stats = asdict(monitor.artifact_stats)
```

For a padded batch, pass a tensor of shape `(B, L, |AP|)` and a length tensor
of shape `(B,)`. Ended traces are frozen while the remaining batch cells are
processed. The public probabilistic API rejects non-finite values and values
outside `[0, 1]`; it also requires factored mode. Inputs may use any floating
PyTorch dtype, and the returned score remains connected to the input through
autograd, including when it is copied to the monitor device.

`artifact_stats` reports:

- the selected representation and device;
- `|AP|`, `2^|AP|`, `|Q|`, transition count, and accepting-state count;
- actual dense-tensor elements and bytes in dense mode;
- cube count, mask elements, and actual cube-tensor bytes in factored mode.

The byte counts cover persistent transition tensors, not Python object
overhead or per-forward temporary tensors. In particular, factored evaluation
temporarily materializes a `(B, C, |AP|)` factor tensor.

## Reproduction checks

From the repository root, with the environment in `environment.yml` active:

```bash
pytest -q tests/test_deep_dfa.py tests/test_deep_dfa_scan.py
```

The tests establish crisp agreement with the symbolic DFA, dense/factored and
sequential/batched/scan agreement, exact WMC against exhaustive valuation
enumeration, row-stochasticity, padded-batch semantics, autograd preservation,
input validation, and consistency of the reported artifact statistics.

The full repository check is:

```bash
pytest -q
```

Compilation uses `ltlf2dfa` and MONA. Dense storage is exponential in `|AP|`.
The cube representation avoids unconditional alphabet materialization but can
also be exponential in the worst case; consult the reported cube count before
using it on large, unfamiliar guards.

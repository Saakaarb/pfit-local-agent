# pfit-claude JAX Reference

This is the compact `pfit jax` reference excerpt copied from
`docs/pfit_claude_reference_packet.md`.

## Boundary

- The LLM translates `generated/user_model.py` into small JAX fragments.
- The deterministic framework owns YAML parsing, CSV loading, parameter scaling,
  Diffrax integration, optimizer loops, result files, plots, snapshots, and
  diagnostics.
- Do not generate framework plumbing such as imports, solver setup, Optax loops,
  file IO, plotting, run-store code, or snapshots.

## user_model.py Contract

`user_model.py` is numpy-style pseudocode. It is the source of scientific logic
for JAX translation.

Required functions:

```python
def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval)
def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters)
def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters)
```

Optional helper for derived observables:

```python
def _observables(solution, trainable_parameters, fixed_parameters)
```

Conventions:

- `trainable_parameters` and `fixed_parameters` are dicts in pseudocode.
- `dataset[:, 0]` is CSV column 1 because time was removed.
- `user_defined_system` returns derivatives for every state in YAML state order.
- `_compute_loss_problem` returns one scalar for one experiment.
- `writeout_description` returns `[time | measured columns | simulated columns]`.
- Missing data must be masked before arithmetic that could produce NaN/Inf.

## Fragment Contract

Return only this JSON object:

```json
{
  "rhs": [],
  "helper_functions": [],
  "loss_body": "",
  "writeout_body": "",
  "review": ""
}
```

Fragment responsibilities:

- `rhs`: one JAX expression per integrated variable, in YAML order.
- `helper_functions`: one complete helper function per entry. Every dependency
  must be passed as an explicit argument.
- `loss_body`: JAX-compatible body inserted into the framework loss function.
  It must return a scalar.
- `writeout_body`: JAX-compatible body inserted into the framework writeout
  function. It must return a 2D array.

Allowed names inside fragments:

- `t`, `solution_time`, `solution`, `dataset`, `trainable_parameters`,
  `fixed_parameters`, `jnp`;
- named states, trainable parameters, fixed parameters;
- helper functions listed in `helper_functions`.

Do not access `constants`, `other_args`, `trainable_variables`, or `diffrax`.

## NumPy to JAX Mapping

| pseudocode | JAX fragment |
|---|---|
| `np.array`, `np.stack`, `np.concatenate` | `jnp.array`, `jnp.stack`, `jnp.concatenate` |
| `np.zeros`, `arr[i] = x` | `jnp.zeros`, `arr = arr.at[i].set(x)` |
| `np.exp/log/log10/sqrt/square/abs` | `jnp.exp/log/log10/sqrt/square/abs` |
| `np.mean/sum/max/min` | `jnp.mean/sum/max/min` |
| `np.isfinite/isnan/nanmax` | `jnp.isfinite/isnan/jnp.nanmax` |
| `np.where`, `np.maximum/minimum/clip` | `jnp.where`, `jnp.maximum/minimum/clip` |
| `np.interp` | `jnp.interp` |
| `np.sign`, `np.tanh` | `jnp.sign`, `jnp.tanh` |

JAX gotchas:

- No Python `if` on traced values; use `jnp.where`.
- No dynamic boolean-mask indexing; use mask arithmetic and `mask.sum()`.
- Prefer vectorized expressions. Avoid dynamic loops and list appends.
- Sanitize invalid values before arithmetic. `jnp.where` may trace both branches.

NaN-safe residual pattern:

```python
mask = jnp.isfinite(data) & jnp.isfinite(scale) & (scale > 0.0)
data_safe = jnp.where(mask, data, 0.0)
scale_safe = jnp.where(mask, scale, 1.0)
resid = jnp.where(mask, (model - data_safe) / scale_safe, 0.0)
count = jnp.sum(mask)
loss_value = jnp.where(
    count > 0,
    jnp.sqrt(jnp.sum(resid * resid) / count),
    1.0e12,
)
```

When returning from `loss_body`, use `return loss_value`. Do not reference
`constants`; the framework provides the surrounding solve-failure handling.

Writeout pattern:

```python
Nts = solution_time.shape[0]
out = jnp.zeros((Nts, 5))
out = out.at[:, 0].set(solution_time)
out = out.at[:, 1].set(dataset[:, 0])
out = out.at[:, 2].set(solution[:, 0])
return out
```

## Runtime API Surface

Generated fragments may use:

- `jax.numpy` as `jnp`: reductions, basic math, finite checks, array
  construction, `where`, `.at`, `interp`, `sign`, `maximum`, `minimum`, `clip`,
  and `tanh`.
- Helper functions from `user_model.py`, translated to explicit-argument JAX.

The deterministic framework uses Diffrax and Optax. The fragments should not
generate Diffrax solver code, Optax optimizer code, or population-search code.

## Loss and Optimizer Behavior

- The framework averages scalar losses across experiments.
- The framework does not normalize the loss; the user/model loss must do that.
- Failed solves map to the framework error loss outside the fragment.
- Avoid hard event extraction in differentiable losses, such as `argmax`, first
  threshold crossing, or hard trigger time.

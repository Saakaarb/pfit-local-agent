# pfit-claude Reference Packet for a Local LLM Agent

Purpose: preserve `pfit-claude` behavior in a local Ollama-based workflow
without copying implementation wholesale. Keep the local implementation minimal:
YAML + numpy-style model pseudocode + generated JAX.

Core boundary:

- LLM owns: study extraction, `user_input.yaml`, `user_model.py`, JAX
  translation, explanations, recommendations, and safe repair suggestions.
- Deterministic framework owns: YAML parsing, CSV loading, validation tools,
  optimizer loops, JAX/Diffrax/Optax execution, run snapshots, source stamps,
  result files, plots, and diagnostics.

## 1. Workflow

```text
pfit new -> pfit check -> pfit jax -> pfit run -> pfit diagnose
```

### `pfit new`

Creates `inputs/user_input.yaml` and `generated/user_model.py` together.

Must collect equations, states, initial conditions, trainable/fixed parameters,
parameter ranges, datasets, column meanings, observables, and loss. It declares
every CSV column in `experiments[].columns`, proposes a normalized loss when the
source did not specify one, chooses `gradient_opt.integrator` explicitly from
RHS/data/bounds evidence, and asks only targeted questions for missing or
ambiguous scientific facts.

Never invent equations, datasets, initial conditions, or unknown parameter
ranges. Initial conditions are not fitted.

### `pfit check`

Validates YAML, model pseudocode, and datasets before JAX translation.

Runs:

```bash
./venv/bin/python3 tools/check_dataset.py <session>
```

Writes `generated/user_input_check.txt`. May auto-correct safe critical errors
in YAML/model, then revalidate up to three times. Never edits CSVs.

Recommendations require user approval. Always emits an R1 integrator verdict:
"kept because..." or "change to ... because...".

### `pfit jax`

Translates `generated/user_model.py` into `generated/generated_script.py`.

Must output pure Python code, verify import, repair up to three times, and stamp:

```bash
./venv/bin/python3 tools/stamp_script.py <session> --write
```

The fit imports `generated_script.py` and never reads `user_model.py`; the stamp
is the stale-translation guard.

### `pfit run`

Full run:

```bash
./venv/bin/python3 tools/check_ready.py <session> --mode full
./venv/bin/python3 fit_parameters.py <session> --live-web
```

Gradient-only:

```bash
./venv/bin/python3 tools/check_ready.py <session> --mode gradient-only
./venv/bin/python3 fit_gradient_only.py <session> --live-web
```

Full fit is required after changes to model, bounds, logscale, integrator, or
population settings. Gradient-only is valid only for gradient-setting changes
when a prior `final_design_point.csv` exists. Do not run with blocking readiness
failures. Generate and inspect fit plots before declaring completion.

### `pfit diagnose`

Diagnoses one completed run and writes `outputs/<run_id>/fit_diagnosis.txt`.
Reads only that run and its snapshot. May edit working session files only after
user approval; never edits snapshots.

If gradient refinement fails with JAX/Diffrax/Equinox/Lineax/NaN/non-finite
gradient errors, run:

```bash
./venv/bin/python3 tools/autodiff_diagnose.py <session> --run <run_id>
```

## 2. Session and YAML

Layout:

```text
sessions/<session>/
  inputs/user_input.yaml
  inputs/*.csv
  generated/user_model.py
  generated/user_input_check.txt
  generated/generated_script.py
  outputs/<run_id>/...
```

Minimal YAML shape:

```yaml
experiments:
  - data_file: data.csv
    columns:
      - {name: time, units: s}
      - {name: X, units: au, observes: X}

model:
  trainable_parameters:
    - {name: k, min_val: 1.0e-03, max_val: 1.0e+02, logscale: true}
  fixed_parameters:
    - {name: R, value: 8.314}
  observables:
    - {name: derived_signal}
  integrated_variables:
    - {name: X, init_val: 1.0}
    - {name: hidden, init_val: 0.0}

population_opt:
  population_size: 100
  num_iters: 20
  processors: 4
  algorithm: DE
  random_seed: 0
  stepsize_rtol: 1.0e-05
  stepsize_atol: 1.0e-07

gradient_opt:
  num_iters: 1000
  stepsize_rtol: 1.0e-07
  stepsize_atol: 1.0e-09
  initial_timestep: 1.0e-06
  initial_time: 0.0
  max_steps: 10000
  integrator: Kvaerno5
  gradient_optimizer: adam
  init_value_lr: 5.0e-03
  end_value_lr: 1.0e-05
  transition_steps_lr: 100
  decay_rate_lr: 0.9

output:
  write_results: true
```

Rules:

- `trainable_parameters` order is the optimization-vector order.
- `integrated_variables` order is the `y[]` state-vector order.
- `fixed_parameters` are constants, not fitted.
- `experiments[].initial_conditions` may override state `init_val` per dataset.
- First CSV column is time. `dataset` passed to model code excludes time.
- `observes` points to either a state or a declared derived observable.
- `uncertainty_of` marks a standard-deviation column.
- Multiple experiments share one trainable vector. Framework calls the loss once
  per experiment and averages scalar losses; user code must not loop over
  experiments.

YAML traps: prefer `1.0e-07`; avoid unquoted names like `on`, `off`, `yes`,
`no`, `true`, `false`.

## 3. `user_model.py` Contract

`user_model.py` is numpy-style pseudocode, not executed by the fit.

Required:

```python
def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval)
def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters)
def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters)
```

Optional for derived observables:

```python
def _observables(solution, trainable_parameters, fixed_parameters)
```

Conventions:

- `trainable_parameters` and `fixed_parameters` are dicts in pseudocode.
- `dataset[:, 0]` is CSV column 1 because time was removed.
- `user_defined_system` returns derivatives for every state in state order.
- `_compute_loss_problem` returns one scalar for one experiment.
- Loss should be normalized, often peak-normalized RMSE or sigma-weighted RMSE.
- `writeout_description` returns `[time | measured columns | simulated columns]`.
- Missing data must be masked before arithmetic that could produce NaN/Inf.

## 4. JAX Translation Contract

Generated structure:

```python
import jax
import jax.numpy as jnp
import diffrax
from diffrax import RESULTS

jax.config.update("jax_enable_x64", True)

@jax.jit
def unscale_value(...)
@jax.jit
def scale_value(...)
@jax.jit
def user_defined_system(t, y, other_args)
@jax.jit
def _integrate_system(constants, trainable_variables)
@jax.jit
def _compute_loss_problem(constants, trainable_variables)
def _write_problem_result(constants, trainable_variables)
```

Framework-owned generated code:

- imports, x64 enable, `scale_value`, `unscale_value`;
- `_integrate_system` shape;
- `diffrax.ODETerm(user_defined_system)`;
- `diffrax.SaveAt(ts=t_eval)`;
- `diffrax.diffeqsolve(..., throw=False, stepsize_controller=PIDController(...))`;
- failure mask:

```python
failed = jnp.invert(result == RESULTS.successful)
loss = jnp.where(failed, constants["error_loss"], loss_value)
```

Only two `_integrate_system` substitutions:

- literal `solver = diffrax.<Integrator>()`;
- literal `max_steps=<int>`.

Do not generate optimizer loops, Optax updates, population-search code,
snapshots, dynamic solver names, dynamic `max_steps`, `throw=True`, or custom
adjoints.

Parameter mapping:

```python
min_val = constants["min_limits"]
max_val = constants["max_limits"]
is_logscale = constants["is_logscale"]
k1, k2 = unscale_value(trainable_variables, min_val, max_val, is_logscale)
R = constants["fixed_parameters"]["R"]
```

Numpy-to-JAX essentials:

| pseudocode | generated JAX |
|---|---|
| `np.array`, `np.stack`, `np.concatenate` | `jnp.array`, `jnp.stack`, `jnp.concatenate` |
| `np.zeros`, `arr[i] = x` | `jnp.zeros`, `arr = arr.at[i].set(x)` |
| `np.exp/log/log10/sqrt/square/abs` | `jnp.exp/log/log10/sqrt/square/abs` |
| `np.mean/sum/max/min` | `jnp.mean/sum/max/min` |
| `np.isfinite/isnan/nanmax` | `jnp.isfinite/isnan/nanmax` |
| `np.where`, `np.maximum/minimum/clip` | `jnp.where`, `jnp.maximum/minimum/clip` |
| `np.interp` | `jnp.interp` |
| `np.sign`, `np.tanh` | `jnp.sign`, `jnp.tanh` |

JAX gotchas:

- No Python `if` on traced values; use `jnp.where`.
- No dynamic boolean-mask indexing; use mask arithmetic and `mask.sum()`.
- Sanitize invalid values before arithmetic. `jnp.where` can still trace both
  branches and poison reverse-mode autodiff.
- Prefer vectorized expressions. Avoid dynamic loops and list appends.

NaN-safe residual:

```python
mask = jnp.isfinite(data) & jnp.isfinite(scale) & (scale > 0.0)
data_safe = jnp.where(mask, data, 0.0)
scale_safe = jnp.where(mask, scale, 1.0)
resid = jnp.where(mask, (model - data_safe) / scale_safe, 0.0)
count = jnp.sum(mask)
loss_value = jnp.where(count > 0,
                       jnp.sqrt(jnp.sum(resid * resid) / count),
                       constants["error_loss"])
```

Writeout:

```python
Nts = solution_time.shape[0]
out = jnp.zeros((Nts, 5))
out = out.at[:, 0].set(solution_time)
out = out.at[:, 1].set(dataset[:, 0])
out = out.at[:, 2].set(solution[:, 0])
return out
```

## 5. Runtime API Surface

Generated code may use:

- JAX: `jax.jit`, `jax.config.update("jax_enable_x64", True)`.
- `jax.numpy`: reductions, basic math, nan checks, array construction,
  `where`, `.at`, `interp`, `sign`, `maximum`, `minimum`, `clip`, `tanh`.
- Diffrax: `ODETerm`, valid solver class, `SaveAt(ts=t_eval)`,
  `diffeqsolve`, `PIDController`, `RESULTS.successful`.

Framework uses, but generated model code should not generate:

- `jax.value_and_grad`, `vmap`, `pmap`;
- Optax optimizer construction/update loops;
- run-store/snapshot code;
- plotting and diagnostics.

Integrator rule:

- Smooth stiff or stiffness unknown -> implicit, usually `Kvaerno5`.
- Smooth non-stiff -> explicit, usually `Dopri5`, `Dopri8`, or `Tsit5`.
- Non-stiff chattering discontinuity -> explicit.
- Stiff plus chattering discontinuity -> fix/smooth model; solver choice alone
  is not enough.
- Hard event losses can break autodiff even when RHS integrator choice is right.

## 6. Optimizer and Loss Behavior

- Population stage: `PSO` or `DE`; `DE` is robust for rugged/non-smooth losses.
- Gradient stage: starts from population best in scaled `[-1, 1]` parameter
  space.
- `logscale: true` means search in log10(parameter).
- New-session default gradient optimizer: `adam`.
- `lbfgs` is for smooth, well-normalized losses expected to match well.
- Framework averages scalar losses across experiments.
- Framework does not normalize the loss; user/model must.
- Failed solves map to `constants["error_loss"]`.
- Avoid differentiating hard event extraction (`argmax`, first threshold
  crossing, hard trigger time). Prefer smooth trajectory residuals or keep event
  scalars out of gradient refinement.

## 7. Validation and Failure Behavior

Blocking examples: missing equations/datasets/initial conditions/ranges,
required YAML fields missing, unknown YAML keys, invalid solver/optimizer,
`initial_time > first data time`, generated JAX import failure, stale generated
script stamp before run.

Common warnings/recommendations: missing explicit integrator, unnormalized loss,
unused uncertainty columns, NaN data without finite-safe masking, many-decade
parameter bounds without logscale, too small population, too few Adam iterations,
row-0 initial-condition mismatch when integration starts at row 0.

Autodiff failure diagnosis checks base loss, `jax.value_and_grad`, finite
differences on scaled parameters, nearby solver walls / `error_loss`, and
non-smooth model/loss constructs.

## 8. Representative Patterns

Simple ODE:

```python
def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
        k1 = trainable_parameters["k1"]
        k2 = trainable_parameters["k2"]
        A, B = y
        return np.array([-k1 * A, k1 * A - k2 * B])
```

Unobserved state: Oregonator integrates `X, Y, Z` but loss compares only `X`
and `Z`. `Y` still needs `init_val`. Structural small parameters `eps1`, `eps2`
make `Kvaerno5` appropriate.

Derived observable:

```python
def _observables(solution, trainable_parameters, fixed_parameters):
        O = solution[:, 0]
        A = solution[:, 4]
        return {"Po": (0.9 * A + 0.1 * O) ** 4}
```

Fixed parameter:

```python
R = fixed_parameters["R"]
r1 = A1 * np.exp(-E1 / (R * T)) * F
```

Helper function: translate helpers by replacing `np` with `jnp` or inline them.
Preserve one definition of shared observable/rate algebra so loss and writeout
cannot drift.

Boehm/STAT5: smooth mass-action model with derived percentage observables and
broad rate spans -> `Kvaerno5`; writeout uses `.at[:, i].set(...)`.

Fujita EGF: multi-experiment shared parameters; experiment-specific stimulus
values are zero-derivative states overridden in `initial_conditions`.


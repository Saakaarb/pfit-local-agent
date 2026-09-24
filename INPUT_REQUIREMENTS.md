# Input Requirements

This is the running input contract for the local-agent pfit workflow. Keep this file updated whenever a new rule is discovered during examples or debugging. These requirements should eventually be folded into the project writeup.

The intended user workflow is:

```text
pfit new <session_dir>
pfit check <session_dir>
pfit jax <session_dir>
pfit run <session_dir>
```

The framework should assume the user starts with only a session directory containing an `inputs/` folder, a data CSV, and a natural mathematical writeup. YAML, Python model files, and JAX scripts are generated artifacts.

## Session Layout

Each fitting problem should live in:

```text
sessions/<session_name>/
  inputs/
    user_info.txt
    <data>.csv
```

Required input files:

- `inputs/user_info.txt`: the user's mathematical/model description.
- One primary CSV data file referenced by name in `user_info.txt`.

Generated files:

- `inputs/user_input.yaml`
- `generated/user_model.py`
- `generated/user_input_check.txt`
- `generated/generated_script.py`
- `outputs/run_<timestamp>/...`

Users should not be expected to write generated YAML or Python by hand.

## CSV Requirements

CSV requirements are strict:

- The CSV must have a header row.
- The first column must be named exactly `time`.
- Every non-time data column header must exactly match either:
  - an integrated state name, or
  - a declared observable name.
- A column header must not be a legacy alias.
- A column header must not require the model to infer what it means from prose.
- A column header must not be reused for multiple simulated quantities.
- Uncertainty columns are allowed only if explicitly described as uncertainty columns.
- If an uncertainty column is present, the prompt must state which measured quantity it belongs to.

Good examples:

```csv
time,T,dTdt
```

```csv
time,C_plasma
```

```csv
time,pSTAT5A,pSTAT5B,rSTAT5A
```

Bad examples:

```csv
time,A_gut
```

when the measured quantity is actually plasma concentration and `A_gut` is an integrated state.

```csv
time,M
```

when the measured quantity is actually `Mpp`.

## User Prompt Requirements

The prompt in `inputs/user_info.txt` should describe the problem mathematically and explicitly.

It should include:

- Dataset filename.
- CSV column meanings.
- Integrated state names and initial values.
- Trainable parameter names, bounds, and logscale flags.
- Fixed parameter names and values.
- ODE right-hand sides.
- Derived observables, if any.
- Loss definition.
- Any required transforms, if user-specified.
- Any known stiffness or integrator requirements.
- Output/writeout expectations if they matter.

The prompt should use names consistently. A name used in an ODE, observable, loss, or parameter list should refer to the same quantity everywhere.

If the prompt includes an explicit `Fixed parameters:`, `Fixed constants:`, or `Constants:` section, each listed value is authoritative. `pfit-new` must preserve those fixed parameter values exactly and fail validation if generated YAML changes or omits them.

If the prompt includes explicit initial conditions using `x(0) = value`, `x0 = value`, or an initial-condition/state section, generated state initial values must match those declarations.

## State Requirements

Integrated states must be explicitly listed.

Each state must have:

- a name,
- an initial value,
- exactly one RHS expression.

Example:

```text
States:
- A_gut(0) = 320.0
- A_plasma(0) = 0.0

ODE:
- dA_gut/dt = -ka * A_gut
- dA_plasma/dt = ka * A_gut - ke * A_plasma
```

State names should be valid Python identifiers. Avoid spaces, hyphens, subscripts, Greek symbols, and unit annotations inside names.

## Observable Requirements

Measured quantities may be states or derived observables.

If a CSV column measures a state, the CSV header must match that state exactly.

If a CSV column measures a derived quantity, the CSV header must match the observable name exactly, and the prompt must define the observable formula.

Example:

```text
CSV:
- C_plasma = observed plasma concentration

States:
- A_gut
- A_plasma

Derived observable:
- C_plasma = A_plasma / Vd
```

Derived observables do not need to be integrated states.

The framework should reject a CSV column if it cannot be reconciled exactly with a state or observable name.

## Parameter Requirements

Trainable parameters must include:

- name,
- lower bound,
- upper bound,
- whether to search on log scale.

Example:

```text
Trainable parameters:
- ka in [0.1, 10.0], logscale true
- ke in [0.01, 1.0], logscale true
- Vd in [5.0, 150.0], logscale true
```

Fixed parameters must include:

- name,
- value.

Example:

```text
Fixed parameters:
- kb = 1.38e-23
```

The model must preserve numeric parameter values exactly. This is a known weak point for large prompts and must be checked.

Initial conditions are not fixed parameters. They belong in the state list.

## Equation Requirements

ODE expressions should be written as mathematical expressions using the declared names.

Allowed math intent:

- addition, subtraction, multiplication, division, powers,
- `exp`, `log`, `log10`, `sqrt`, `abs`,
- smooth functions such as `tanh` or sigmoid when needed.

Preferred prompt style:

```text
- r1(c1,T) = -A1 * exp(-Ea1 / (kb*T)) * c1
- dc1/dt = r1(c1,T)
```

The local workflow currently expects generated Python expressions to use `np.*` in `user_model.py` and `jnp.*` in generated JAX. The user prompt itself can use normal mathematical notation.

## Conditional And Switch Requirements

Hard Python conditionals are not allowed in generated differentiable dynamics.

Avoid:

```python
if x > 0:
    ...
```

Avoid:

```python
a if condition else b
```

Use smooth or vectorized alternatives:

- sigmoid switch,
- `tanh` switch,
- `np.where` or `jnp.where` only when appropriate,
- `np.logical_and` / `jnp.logical_and` for boolean combinations.

If a smooth switch is required, every constant in the switch must be defined.

Bad:

```text
smooth_switch(x, threshold, width)
```

without defining `threshold` and `width`.

Good:

```text
S(z) = 1 / (1 + exp(-1000*z))
```

or:

```text
Use 0.5 * (1 + tanh((x - threshold) / 0.01)).
```

The smoothing width must be either:

- a literal number, or
- a declared fixed parameter.

## Helper Function Requirements

Helper functions are a current weak point.

Until helper-function support is formalized, prefer fully inlined expressions in the prompt.

Avoid requiring the model to invent helper functions such as:

```text
sign(x)
smooth_switch(x, threshold, width)
rate_law(...)
```

If helper functions are unavoidable, the prompt must define:

- function name,
- arguments,
- expression body,
- all constants used inside it.

Generated helper functions must not close over undeclared constants. Every nonlocal value used by a helper must be passed as an argument or declared as a parameter/fixed parameter.

The framework should eventually support helper functions explicitly, but right now inline expressions are safer.

## Loss Requirements

The prompt should state the loss clearly.

If the user provides a loss, the framework must preserve that loss. It should not replace it with a default deterministic shortcut.

`pfit-check` performs deterministic checks for explicit loss contracts. If the prompt asks for RMSE/square-root loss, log/log10 residuals, or normalized/scaled residuals, `_compute_loss_problem` must contain the corresponding operation.

When a prompt asks for normalization by `max(abs(measured column))`, generated loss code should use max-absolute normalization rather than range normalization.

A loss description should specify:

- which simulated quantity is compared to which measured column,
- whether residuals are normalized,
- the normalization scale,
- whether the final data loss is MSE or RMSE,
- whether the comparison is linear, log, or log10,
- any penalties,
- whether penalties are smooth/differentiable.

Example:

```text
Loss:
- Compare simulated C_plasma to measured C_plasma.
- Normalize by max(abs(measured C_plasma)).
- loss = sqrt(mean(normalized squared residuals)).
```

If the user does not provide a loss, the framework may choose a default normalized MSE.

If the measured quantity is positive and spans multiple orders of magnitude, a log/log10 residual should be considered. `pfit-check` should catch cases where this is required.

For heat-rate or rate-like data spanning orders of magnitude:

```text
Compare heat rate in log10 space before normalization.
```

## Penalty Requirements

Penalty terms must be differentiable when used in JAX optimization.

Avoid hard if/else penalties.

Use smooth penalties such as:

```text
S(z) = 1 / (1 + exp(-sharpness*z))
```

Every penalty must specify:

- quantity being penalized,
- threshold or target,
- penalty scale,
- sharpness if using a sigmoid,
- whether the penalty applies at all times or final time only.

## Stiffness And Integrator Requirements

If a model is stiff, the prompt should say so.

Example:

```text
This is a stiff chemical kinetics model. Use Kvaerno5 for gradient integration.
```

The framework currently attempts to infer stiffness from context and parameter ranges, but explicit stiffness notes are safer.

Known stiff examples:

- Robertson kinetics.
- Boehm STAT5.
- ARC thermal runaway.
- Some Oregonator regimes.

## Large Model Requirements

Very large biochemical systems are currently unreliable with a single broad dynamics extraction.

For large systems, the prompt should be especially structured:

- list all fixed parameters separately,
- list all trainable parameters separately,
- list all states and initial values separately,
- list reaction rates separately,
- list RHS equations separately,
- list observables separately,
- list loss separately.

Large systems should avoid prose-only descriptions.

Known failure case:

- `nfkb_signaling`

Observed NF-kB failures:

- fixed parameter value changed,
- states emitted without initial values,
- equations simplified/invented,
- invalid JSON shape.

This indicates that large systems need more pfit-new substeps and stronger deterministic cross-checks.

Current implementation adds a large-model equation-extraction mode when the state/parameter count or prompt size is high. This mode freezes state and parameter names in the equation prompt and tells the LLM to return `missing_inputs` rather than simplifying or inventing equations.

## Validation Expectations

The framework should fail early when:

- a CSV header does not exactly match a state or observable,
- a state lacks an initial value,
- an observed column maps to more than one simulated quantity,
- a required parameter is missing,
- a fixed parameter value differs from the prompt,
- an RHS references an undefined name,
- an observable references an undefined name,
- a loss references an undefined name,
- a helper constant such as `width` is not declared,
- generated JSON does not match the expected schema,
- generated equations invent names not present in the prompt.

## Known Current Gaps

These are not input requirements for users; they are framework deficiencies discovered during testing.

- `pfit-check` now catches simple deterministic custom-loss mismatches, but complex algebraic equivalence is still an LLM/semantic-review task.
- Large prompts can still cause the local model to simplify or invent equations, though fixed parameter value drift is now rejected when values are declared explicitly.
- Large prompts can cause the local model to invent simplified equations.
- Helper functions are not represented robustly.
- Targeted repair can fix syntax while introducing undefined constants.
- Successful workflow generation does not guarantee good fit quality under tiny optimization budgets.
- Some generated JAX fragments are overlong but recoverable by validation/assembly.

## Current Example Status

Examples that currently pass through `pfit-new -> pfit-check -> pfit-jax`:

- `theophylline`
- `lotka_volterra`
- `robertson_session`
- `mapk_cascade`
- `vanderpol_session`
- `oregonator`
- `piezo_bouc_wen`
- `boehm_stat5`
- `ARC_fitting`

Examples currently failing at `pfit-new`:

- `sliding_basepoint_headered`
- `nfkb_signaling`

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

The prompt in `inputs/user_info.txt` must describe the problem mathematically and explicitly.

It should include:

- Dataset filename.
- CSV column meanings.
- Integrated state names and initial values.
- Trainable parameter names, bounds, and logscale flags.
- Fixed parameter names and values.
- ODE right-hand sides.
- Derived observables, if any.
- A mathematical loss/objective definition. This is required; the framework
  must not be asked to invent or select the scientific objective.
- Any required transforms, if user-specified.
- Any known stiffness or integrator requirements.
- Output/writeout expectations if they matter.

The prompt should use names consistently. A name used in an ODE, observable, loss, or parameter list should refer to the same quantity everywhere.

If the prompt includes an explicit `Fixed parameters:`, `Fixed constants:`, or `Constants:` section, each listed value is authoritative. `pfit-new` must preserve those fixed parameter values exactly and fail validation if generated YAML changes or omits them.

If the prompt includes explicit initial conditions using `x(0) = value`, `x0 = value`, or an initial-condition/state section, generated state initial values must match those declarations.

Unambiguous group initial conditions are accepted: for example, “A starts at 1
and all other reaction states start at zero” supplies values for every named
member of that group. Individual declarations and experiment overrides take
precedence. Filename-keyed tables may supply clamp values for zero-derivative
states; no separate global value is needed when every experiment supplies one.
Before reporting missing state values, extraction makes one recheck of the
original specification when repair attempts are enabled. Truly missing or
ambiguous values still require user input; unspecified values are not assumed zero.

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

The prompt must state the loss mathematically and unambiguously. A prose request
such as "fit the data" or "minimize the error" is not sufficient for a new
problem setup.

The framework must preserve the supplied loss. It must not replace it with a
default deterministic shortcut.

`pfit-check` performs deterministic checks for explicit loss contracts. If the prompt asks for RMSE/square-root loss, log/log10 residuals, or normalized/scaled residuals, `_compute_loss_problem` must contain the corresponding operation.

Semantic review uses the same runtime column/state mapping as JAX translation:
time is removed before the model is called, so `dataset[:, 0]` is the first
measured CSV column and `solution_time` holds time. `solution` columns follow
integrated-variable order. Review receives the stated loss, literal division
expressions from its source, and actual per-column NaN/infinity counts. Aliases,
helpers, derived observables, and vectorized losses remain supported; expression
evidence is not a complete proof of equivalence. Critical semantic findings must
quote an offending source expression and explain a contradiction with these
facts or the stated loss. Uncertain or hypothetical concerns are warnings.

Only deterministic validation findings block `pfit check`. LLM-reported critical
findings are retained as `Unverified semantic finding` warnings, not silently
discarded or treated as proof. A passing check means deterministic preflight
passed; it does not certify scientific correctness, and semantic warnings still
need review. JAX translation continues to require deterministic validation and
smoke tests. The checker recognizes direct or straight-line aliased NumPy/JAX
max/min/range reductions of simulated states or time used as loss denominators
when measured-data normalization is explicitly requested, and rejects them.
Unknown helper algebra remains a semantic-review concern rather than a claimed
proof of correctness or failure.

When a prompt asks for normalization by `max(abs(measured column))`, generated loss code should use max-absolute normalization rather than range normalization.

A loss description should specify:

- which simulated quantity is compared to which measured column,
- whether residuals are normalized,
- the normalization scale,
- whether the final data loss is MSE or RMSE,
- whether the comparison is linear, log, or log10,
- any penalties,
- whether penalties are smooth/differentiable,
- how multiple measured columns are combined,
- how multiple experiment records are combined, and
- any uncertainty weighting or other per-point weights.

The definition should include an equation or equivalent explicit mathematical
expression. Every symbol in it must map to a named simulated quantity, measured
CSV column, parameter, or stated constant. It must also identify the reduction
domain: which time points, columns, and experiment records contribute to each
mean, sum, or root operation. If data can be missing, it must state whether the
reduction is over finite observed entries only and what should happen when a
required channel has no finite observations.

Example:

```text
Loss:
- Compare simulated C_plasma to measured C_plasma.
- Normalize by max(abs(measured C_plasma)).
- loss = sqrt(mean(normalized squared residuals)).
```

For new inputs, omission of the loss is an incomplete problem specification.
Existing automatic/default-loss behavior is retained only for compatibility
with older prepared sessions and must not be relied on in new tutorial examples.

An explicit user loss definition takes precedence over automatic data-scale
heuristics in both `pfit new` and `pfit check`. For example, a request to compare measurements directly with
max-absolute-normalized RMSE remains an original-scale loss even if a column
spans many orders of magnitude. `pfit check` warns about that dynamic range but
does not require a different objective. Parameter `logscale` settings describe
search coordinates, not residual transformations. Explicitly requested RMSE,
log/log10 transformations, normalization, and uncertainty weighting are still
checked against the generated loss.

When no user loss is supplied, the automatic log-loss rule applies independently
within each measured column:
there must be at least two finite values, every finite value must be strictly
positive, and `log10(max/min) >= 3` (a ratio of at least 1,000) within that same
column. In that case, `pfit check` requires log/log10 residuals for that column
before normalization. A generated linear loss is rejected; the checker must not
silently rewrite the user's loss.

Differences in scale between columns, large absolute values, time ranges, and
trainable parameter bounds do not trigger this rule. Neither do positive ranges
below three orders of magnitude. Columns containing zero or negative finite
values are ineligible; do not discard those values or take absolute values to
make a column qualify. Missing/nonfinite values are excluded from the range
calculation and remain subject to separate loss-safety checks. The deterministic
check and semantic-review summary use the same full-column range calculation.
When the automatic rule does not apply, preserve the stated linear loss; an
explicit user request for log/log10 residuals must still be honored.

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

## Gradient-only restarts and sloppiness

Use `pfit run sessions/<name> gradient-only --from-run <run-id>` to refine a saved
physical design point with the current session model and numerical settings.
The source is a run ID or run directory. Without `--from-run`, the latest completed run with a saved point is selected; legacy flat outputs are considered last. `--seed-run` is an alias.
New runs write `final_parameters.json`; parameters are matched by name, so YAML
reordering is supported. The parameter set must match, all seed values must be
finite, and values must lie within the current bounds. Log-scale bounds must be
positive; all bounds must be finite and strictly increasing. Changed bounds or
logscale flags are supported when the physical seed remains valid. Regenerate
JAX after changes affecting generated model code or its parameter order.

Older runs with only `final_design_point.csv` and no saved configuration have no parameter-name metadata.
`--allow-legacy-seed` explicitly asserts that their CSV order matches the current
YAML; otherwise they are rejected. Seeds are never silently clipped. This starts
a fresh gradient optimizer, not a continuation of its internal Adam or L-BFGS state.
Every run, including the Python entry point, writes a new timestamped directory;
an existing explicitly selected output directory is rejected rather than deleted.

Post-fit sloppiness is enabled by default. Use `--no-sloppiness` to skip it or
`--sloppiness-method finite-difference` to bypass second-order autodiff. The
`auto` method tries second-order autodiff and falls back to finite differences
of first-order autodiff gradients. Curvature is measured in physical coordinates
for linear parameters and log10 coordinates for log-scaled parameters, at the
saved best point and gradient-stage tolerances. Nonfinite derivatives or failed
integration make the diagnostic fail separately without discarding fit results.
Nonsmooth losses can therefore yield a usable fit but no usable curvature report.

`sloppiness.json` and `sloppiness_report.txt` report method, coordinates,
eigenvalues/eigenvectors, locally weak modes, stationarity and boundary caveats.
CSV matrices use the parameter order in the JSON; eigenvectors are columns in
descending eigenvalue order. Curvature does not certify convergence or global
identifiability. `pfit diagnose` includes this saved analysis and recognizes the
intentional absence of population-search logs on gradient-only runs.

The deployed 60-parameter sloppiness limit is retained. A successful analysis also
writes `sloppiness_spectrum.png`. `python analyze_fit.py <session> --run <run-id>`
recomputes analysis from a recorded run snapshot without repeating fitting.


Deterministic readiness (2026-09-26)

- One or more experiment records are supported. All records are validated and
  fitted; see the multi-experiment contract below.
- CSV data must contain at least two rows, a finite strictly increasing time
  column, and at least one measurement column. Declared columns must match the
  full CSV width, including auxiliary uncertainty columns. Rows must have equal
  width. Blank/NaN measurements remain accepted; infinite values are rejected.
  Existing headerless sessions remain readable; `pfit new` still requires a header.
- Bounds must be finite with `min_val < max_val`; logarithmic bounds must be
  positive. `logscale` must be a YAML boolean. Fixed values and initial conditions
  must be finite; model names must be Python identifiers.
- Population sizes/processors/max_steps must be positive integers; DE requires
  at least three population members. Iteration counts and random seeds must be
  nonnegative integers. Tolerances must be positive and finite, either scalar
  or one per integrated state. Initial timestep and error_loss must be positive
  and finite. Explicit initial_time must be finite and no later than the first
  measurement time. Tiny budgets remain valid for smoke tests.
- `population_opt.processors` is the maximum number of JAX CPU devices requested
  by the session. At run startup it is capped by CPUs available to the process,
  including affinity limits. An explicit host-device count in `XLA_FLAGS` is an
  advanced override. Resource decisions and the actual JAX device count are
  recorded in the run manifest.
- Run manifests record best-effort environment, package, Git, optimizer/seed,
  configured Ollama, and artifact-hash provenance. Provenance collection is
  informational: missing Git/package/configuration metadata does not reject an
  otherwise valid run. The Ollama fields describe configuration visible at run
  time and may differ from command-line overrides used for earlier generation.
- `pfit check SESSION --deterministic-only` checks inputs without Ollama.
  `pfit check SESSION --ready` additionally checks generated-code presence,
  interface and source freshness. Every `pfit run` repeats these checks; restart
  seed validation also runs before fitting.
- Successful `pfit jax` records hashes of the complete model and YAML contents.
  Any edit to either source (including comments or optimization budgets) requires
  rerunning `pfit jax`. Older unstamped scripts use a timestamp fallback with a
  warning; copies/clones cannot establish content agreement through timestamps.
  Failed newly generated scripts do not receive a valid stamp.

These checks do not prove translation fidelity, gradient correctness or complete
missing-data support. Those remain separate validation/parity work.


Multi-experiment contract (2026-09-26)

Declare one entry per experiment under `experiments`, each with `data_file`,
`columns`, and optional `initial_conditions`. Initial-condition overrides must
be finite values keyed by integrated state names; all other states inherit the
model's global initial values. Equations, parameter values/bounds, fixed
parameters, observables and solver settings are shared across records.

```yaml
experiments:
  - data_file: run_a.csv
    columns:
      - {name: time}
      - {name: y, observes: y}
  - data_file: run_b.csv
    columns:
      - {name: time}
      - {name: y, observes: y}
    initial_conditions: {y: 2.0}
```

Every record must have the same ordered column names, observation mappings,
uncertainty roles and units. Column names must be unique; `uncertainty_of` must
reference a measurement column in that record. CSV headers, when present in a
multi-experiment session, must match those declarations. Existing headerless
reference sessions remain readable, with column meanings supplied by the YAML.
Different row counts, initial times and final times are supported. Without an
explicit `gradient_opt.initial_time`, each solve starts at that record's first
time; an explicit global initial time must be valid for all records.

Generated functions operate on one record at a time. The framework computes the
arithmetic mean of per-record scalar losses, giving each experiment equal weight.
It does not pool observations or replace a mean of RMSEs with pooled RMSE.
A solver failure, nonfinite scalar loss, or error-loss sentinel from any record
invalidates the whole candidate. Newly generated code also rejects nonfinite
simulated trajectories even when a custom loss attempts to mask those values.

Blank/NaN measurements remain present. The standard generated loss masks missing
measurements before normalization and rejects empty observation channels or
nonfinite simulations. Custom/uncertainty losses remain responsible for valid
masking before logs/division; every record must pass the translation smoke test.
Do not assume this port supplies a complete arbitrary-custom-loss masking audit.
For multi-experiment `pfit new`, automatic post-extraction log-loss rewriting is
disabled so the first CSV alone cannot alter the shared objective; extracted loss
intent and checks cover the full experiment context. Explicit user losses retain
precedence.

`pfit new` accepts a structured `experiments` selection (data_file and optional
initial_conditions) while retaining legacy `filename_data` responses. All selected
CSV headers must agree. Supply explicit experiment conditions; the workflow must
not infer state initial conditions from a derived measurement. Unknown experiment
keys, custom experiment weights, differing observation layouts, per-experiment
fixed/fitted parameter overrides and measured forcing histories are unsupported.

Full fitting and gradient-only restarts use every experiment. Each run snapshots
all CSVs and records original filenames, SHA-256 hashes and resolved initial
conditions in run_manifest.json. Multi-record results use result_solution_expN.csv;
single-record results retain result_solution.csv. fit_summary.json records each
experiment's loss and the equal_experiment_mean aggregation rule. Historical
sloppiness uses the complete snapshot, not the current working datasets.

Explicit loss descriptions may use a one-line `Loss: ...` declaration or a
multiline `Loss:` section. End the section with the next colon-terminated heading
before appending reference code. Statements about already normalized observables
in reference comments do not request additional residual scaling. Intermediate
rate definitions must be supplied when equations refer to them; extraction and
repair may inline these definitions but must not invent missing rates.

Explicit plain RMSE is supported without residual normalization. Structured
loss terms and penalties are preserved even if extraction labels them non-custom.

Measured forcing inputs

Declare a known input column as `{name: pump_voltage, role: forcing,
interpolation: linear}`. Its name must be a distinct Python identifier, separate
from states, parameters and observables. A forcing column cannot also declare
`observes` or `uncertainty_of`. It is excluded from the standard fitted-observation
mapping and log-loss rules. Every experiment must use the same column roles and
units; values and sampling times may differ.

Forcing is currently sampled on the CSV time grid and linearly interpolated at
ODE solver times. Every forcing sample must be finite. The CSV must cover the
whole integration interval, including initial_time; extrapolation is rejected.
Separate forcing grids, missing-input imputation, zero-order hold and unit
conversion are not supported. The first and last samples define the supported
interval. Piecewise inputs that require a hold policy must not be silently
converted to linear ramps.

In fresh extraction, equations declare forcing_columns using the CSV headers and
use those names as scalar inputs; interpolation bindings are generated. Prepared
Python models may use np.interp(t, t_eval, dataset[:, i]); the runtime data array
excludes time, so raw CSV column i+1 is runtime column i. JAX translation remains
an LLM step and preserves these inputs as jnp.interp or supplied scalar bindings.

Loss and writeout preservation during JAX translation

A supplied _compute_loss_problem is authoritative, including plain MSE/RMSE,
normalization, transformations and penalties. Functions are not reclassified as
defaults based on their algebraic shape. Only an explicitly marked, unchanged
framework skeleton permits a default substitution. Custom writeout_description
(or legacy write_problem_result) is preserved independently of the loss.

Source model functions must be executable with the documented Python signatures.
Numerical source/JAX probes compare equations, loss and output columns before
acceptance and route mismatches into bounded repair. See
[loss_and_translation_fidelity.md](docs/loss_and_translation_fidelity.md) for
probe scope and tolerances. Scientific correctness of extracted source equations
still requires review; finite numerical probes are not a proof of equivalence.

### Gradient optimizer selection

`gradient_opt.gradient_optimizer` accepts `adam` (default) or `lbfgs`,
case-insensitively. Omitted `gradient_opt.num_iters` defaults to 1000; an explicit
nonnegative integer is preserved, including zero to skip gradient updates.
New sessions write Adam and 1000 iterations explicitly. These are gradient-stage
settings; population-stage budgets are unchanged. Existing YAML without an
optimizer now selects Adam; specify `lbfgs` to reproduce the previous selection.

Adam uses the supplied `init_value_lr`, `end_value_lr`, `transition_steps_lr`,
and `decay_rate_lr` exponential schedule. All four must be positive and finite.
Their defaults remain `1e-4`, `1e-5`, `2000`, and `0.9`. L-BFGS uses its existing
Optax line search and does not use that schedule. Both full fitting and
gradient-only restarts honor the selection; restarts create fresh optimizer
state. The fit summary records the effective optimizer and iteration budget.
A budget of 1000 is a default, not a convergence guarantee.

### Preserving an explicit RMSE request during structured extraction

The split loss-extraction workflow now uses the original `user_info.txt` loss
specification when normalizing extracted metrics. An unambiguous pooled RMSE
request upgrades supported MSE variants to their RMSE counterparts, even if the
LLM's review or notes omit the square root. The existing renderer averages the
channel mean-square contributions and then applies one square root. Residual
mappings, normalization/uncertainty scales, and penalties are retained. Applied
corrections are recorded in `generated/pfit_new_review.txt`.

This safeguard is deliberately narrow: it does not infer negated, alternative,
mixed-metric, or separate per-channel RMSE objectives. Those still require a
faithful extraction or explicit user review. It does not rewrite supplied Python
loss bodies, add missing-data policies, or implement a check-time repair loop.

### Solver diagnostics and bounded JAX validation recovery

New `pfit new` configurations include:

```yaml
gradient_opt:
  max_steps: 10000
  solver_recovery_max_steps: 50000
  solver_recovery_timeout_seconds: 900
  solver_validation_min_successful: 10
  solver_validation_seed: 7
```

`pfit jax` tests a reproducible parameter sample across the supplied search
bounds. One sample set is chosen before integration and reused unchanged at
every step budget. The default count is `min(128, max(32, 16*d))`, where `d` is
the total number of search axes, including estimated initial conditions. An
explicit `solver_validation_samples` overrides this default with a fixed count.
The legacy `solver_validation_max_samples` is ignored: sample sizes never grow.
The first vector is the midpoint; subsequent vectors are generated in
seeded Latin-hypercube blocks. Coordinates are linear for linear parameters
and logarithmic for log-scaled parameters. The combined design is not a single Latin hypercube.
The count is a bounded heuristic based on dimension, not physical bound volume
(which depends on units) or a guarantee of high-dimensional coverage.

Require **10 successful parameter vectors**, independent of the total sample count
(for example, 32, 80 or 128). `solver_validation_min_successful` is a fixed count, default 10.
Legacy fraction and stagnation settings no longer control readiness. A vector
passes when every experiment integrates successfully with finite outputs and
returns a finite, non-sentinel loss. Completion does not require a low loss.
The midpoint need not pass. Writeout and source/JAX fidelity checks use feasible
sampled vectors; numerical coverage does not replace translation validation.

Per-sample, per-experiment results, step counts, physical/normalized parameters
and finite-output counts are recorded in `generated/solver_diagnostics.json`.
`generated/solver_coverage.json` records completion fractions and failure
categories for every attempt. These are empirical coverage checks, not a
proof that the complete parameter space is feasible or that all useful
solutions fit within the selected step budget. Custom legacy scripts lacking
`_integrate_system_with_stats` retain the older midpoint contract check and
cannot provide this sampled solver-coverage report; regenerate with `pfit jax`
to obtain the generated statistics interface.

If fewer than ten samples succeed and step exhaustion occurred, recovery doubles
`max_steps` up to `solver_recovery_max_steps`. It reuses the exact same parameter
vectors and seed; it never adds points. Without step exhaustion there is no
step-budget retry. Equations, solver, reference tolerances, initial conditions,
losses and data remain unchanged. No per-particle step limits or runtime fitting
retries are introduced.

Recovery stops when the required successful count is met or step/time
limits are exhausted. There is no fraction-based stagnation stop. The success
fraction remains diagnostic only. A small feasible region is not rejected merely
for occupying a small percentage of the search space. Ten successful points do
not guarantee coverage of all useful solutions or an adequate fit.

The recovery timeout defaults to 900 seconds (15 minutes) and bounds the entire additional-recovery phase, including
subprocess startup, compilation and fidelity checks. It does not time-limit
the initial ordinary validation attempt. Larger/multi-experiment validation
sets may need a larger timeout; a timeout does not establish stagnation.
Existing/custom YAML without `solver_recovery_max_steps` retains its configured
`max_steps` as the hard ceiling. New sampling fields use the defaults above
when omitted. Sample counts and the required-success count must be positive
integers; the selected fixed sample count must be at least the required
success count. The seed must be a nonnegative integer. The step ceiling must be an
integer at least as large as `max_steps`, and timeout positive and finite.

After successful recovery, accepted step budgets are saved to YAML.
The script receives a fresh source/config stamp only after numerical, contract
and source/JAX fidelity checks pass. On exhausted recovery the original
configuration is restored and the script remains marked pending. Concurrent
source edits are not overwritten or stamped as validated. Recovery history is
saved in `generated/solver_recovery.json`.

Numerical failures do not invoke LLM code repair. Translation repairs are limited
to the fragment field implicated by the validation error. Deterministically
translated source loss/writeout bodies are protected; unrelated changes are
rejected, and repeated identical failed proposals stop early. A scientific
fidelity mismatch remains a failure rather than being bypassed. These rules
apply to JAX translation only; the check-stage repair loop remains deferred.

Gradient refinement reports non-finite loss or gradients as an early stop, names
affected gradient parameters, and retains the best valid fitting point. The
console and `NODE_fitting.log` report the warning; `fit_summary.json` records
`termination` and `termination_detail`. A finite retained fit does not establish
that gradient refinement completed successfully.

### Refinement-first tolerance selection

Numerical readiness first selects `max_steps` using the configured gradient-stage
(refinement) tolerances. After coverage and source/JAX fidelity pass, it tests
state-scaled absolute tolerances, then a 10x DE relaxation, with the same fixed
step ceiling throughout. Each comparison evaluates every successful sampled
parameter vector across all experiments; at least two successful vectors are
required. No additional parameter search or step-budget escalation is performed
for tolerance selection.

For each successful trajectory, the scale of each state is the larger of its
95th-percentile absolute value at saved observation times and its absolute
initial value. Across parameter candidates, the median scale is taken separately
for each experiment; the largest of these experiment scales becomes `S_i`.
This gives experiments equal weight regardless of their number of observations
and avoids letting one extreme parameter sample determine the scale. Values
between saved times, including narrow unsampled peaks, are not measured.

The proposed refinement tolerance is `atol_i = rtol_i * 0.01 * S_i`; refinement
`rtol` is unchanged. This puts the crossover between absolute and relative error
control at 1% of each characteristic state scale. The proposal may tighten or
loosen individual states' tolerances. A zero scale or a nonpositive/nonfinite
computed tolerance retains the configured absolute tolerance for that state.
Missing trajectory-scale information skips state scaling. Set
`gradient_opt.auto_state_tolerances: false` to preserve explicitly chosen
refinement tolerances exactly; otherwise this selection is enabled by default,
including for existing YAML that does not specify the flag.

A proposal is accepted only if every paired solve succeeds with valid loss and
each experiment loss stays within `1e-8 + 0.01 * abs(reference_loss)`. Meaningful
ordering of mean candidate losses must also remain unchanged, ignoring reference
gaps within that scale-dependent threshold. A failed state-scaling check retains
all configured refinement tolerances. The DE proposal then multiplies the
selected refinement `rtol` and `atol` by 10. Both comparisons use the original
configured-tolerance losses as reference, so their error allowances do not
accumulate. A failed DE check retains the selected refinement settings for DE.

Selected tolerances are frozen across particles, experiments and fitting
iterations. This is a sampled loss-stability check, not a guarantee of global
trajectory or gradient accuracy. Small-magnitude states may still be important
to the dynamics. Existing numerical failures must pass baseline coverage before
state scaling is attempted; this policy does not rescue a model for which no
sufficiently broad set of successful reference integrations is available.

The decisions, state scales, fallback state indices and paired losses are saved
in `generated/tolerance_calibration.json`. Per-trajectory scales are also saved
in the solver-coverage diagnostics. Accepted refinement `stepsize_atol` and DE
`stepsize_rtol`/`stepsize_atol` are written to their respective YAML sections,
with the source stamp refreshed after validation. Explicit population tolerances
are preserved independently of state scaling; remove them to request automatic
DE selection again. Set `population_opt.auto_tolerances: false` to disable only
the DE comparison. Disable both options to keep all configured tolerances.
Legacy scripts without solver statistics skip both checks. Additional tolerance
checks count against the recovery deadline when run inside bounded recovery.

### Dataset-based integrator selection

Fresh configurations enable `gradient_opt.auto_integrator: true`. Before JAX
translation, the workflow selects one solver using only mapped measurement
columns across all experiments. The initial YAML uses Kvaerno5 as a placeholder.
Existing YAML without this flag, or with it false, preserves its explicit solver.
With the flag true, dataset selection replaces the configured integrator.

For each column with at least eight finite observations, a five-point rolling
median suppresses isolated spikes. Using the corresponding interior time points,
let A be the smoothed peak-to-peak amplitude and v the absolute adjacent slopes.
Exclude numerically zero slopes. Estimate fast and slow times as
`A / percentile90(v)` and `A / percentile10(v)`. Across all measured columns and
experiments, divide the largest slow time by the smallest fast time:

- Ratio >= 100: select Kvaerno5.
- Ratio < 100 with informative, resolved data: select Tsit5.
- Insufficient observations/changes, unresolved fast variation (fast time less
  than two median sample intervals), or no non-flat measurements: select Kvaerno5.

Forcing, uncertainty and auxiliary columns do not enter the calculation. Missing
observations are omitted. Flat columns provide no time-scale evidence. The
100-fold threshold is an explicit heuristic, not a mathematical stiffness test.
Observed curves can hide fast unmeasured or already-decayed modes; no dataset-only
method can certify non-stiffness across the parameter search space. Noise and
sparse sampling can also affect the estimate. Reports expose the evidence and
limitations in `generated/solver_selection.json`.

This is the only automatic solver-selection method. There are no keyword rules,
parameter-bound heuristics, or switches triggered by successful/failed solver
probes. Sampled coverage, bounded step recovery, tolerance calibration,
and forward-accuracy validation still run, using the selected solver throughout.
A validation failure blocks readiness; it does not select another integrator.
Numerical failures do not invoke LLM repair.

### Initial max-step estimate from measured time scales

Fresh configurations enable `gradient_opt.auto_max_steps: true`; absent/false
preserves the configured starting budget. Before translation, reuse the dataset
time-scale analysis. For each experiment with resolved time scales, take its
smallest fast time tau and duration T (last measurement minus initial_time, or
minus first measurement when initial_time is absent). Estimate `100*T/tau`:
ten steps per observed time scale and a factor of ten for headroom. Round up
to the next 1,000, with a minimum of 1,000 and the configured recovery ceiling
(default 50,000). Select the largest experiment budget for all particles.
Insufficient, unresolved or entirely flat measurements retain the configured
initial budget (default 10,000) for that experiment. Flat columns are ignored
when other informative columns exist. This estimate can lower the initial budget.

The multipliers are heuristic, not guarantees or prescribed internal time steps.
The estimated budget and any clipping are recorded under `max_steps_estimate`
in `generated/solver_selection.json`. Existing sampled completion checks,
step increases, recovery deadline and tolerance
validation remain unchanged. Explicit solver selection and automatic budget
estimation can be enabled independently. No fitting proceeds solely on this estimate.

### Parallel forward-accuracy acceptance

Fresh configurations enable `gradient_opt.solver_accuracy_check: true`.
Existing YAML can opt in with that flag; absent/false preserves legacy behaviour
without claiming an accuracy check. After coverage and tolerance proposals, the
workflow checks the exact proposed refinement and population settings against
integrations at 10x tighter relative **and** absolute tolerances. The step
ceiling stays fixed. No proposed tolerance configuration is saved/stamped until
this check passes. If only the automatic DE relaxation fails, DE retains the
already-validated refinement settings; no additional solves are needed. Explicit
population tolerances are not overwritten. A failed refinement check does not
automatically adopt the tighter reference: those settings have not themselves been validated against a further
reference. It blocks readiness without changing the selected solver.

At most four feasible parameter vectors are selected deterministically: the
lowest mean-loss candidate, the candidate with most integration steps across
experiments, and additional points maximizing distance from those already
selected in normalized parameter coordinates. If fewer are feasible, all are
used; the report records the actual probe count. Each probe covers every
experiment and both tolerance profiles. `solver_accuracy_workers` defaults to
4 (allowed 1–4), capped by available CPU affinity and probe count. Independent
probes run concurrently in a shared-process thread pool; experiments and
profiles within a probe run sequentially. This adds bounded validation
parallelism, not parallelism to the fitting stages. Trajectories are reused for
loss evaluation and shared between profiles when their tolerance levels match.
First-use JIT compilation can still dominate a small check.

For every mapped measured state or derived observable at finite observation
rows, require the maximum absolute prediction difference divided by its allowed
error to be at most one. With valid positive measurement uncertainty `sigma`,
the allowed difference is `solver_accuracy_uncertainty_fraction * sigma`
(default 0.01). Otherwise it is
`solver_accuracy_atol_scale * S + solver_accuracy_rtol * abs(reference)`
(defaults 1e-6 and 1e-3), where `S` is the maximum absolute measured/reference
value for that observable in that experiment and probe. These are declared
comparison thresholds, separate from integrator tolerances; they are adjustable
positive finite numbers. An identically zero signal requires exact agreement
when its allowed error is zero. Forcing and auxiliary columns are excluded.
Derived measurements use the generated `_observables` helper; a missing helper
or absence of mapped finite measurements blocks validation explicitly.

Per-experiment losses and meaningful mean-loss ordering must also pass the
existing `1e-8 + 1% * abs(reference_loss)` comparison. Reports are saved in
`generated/solver_accuracy.json`, including selected probes, workers, thresholds,
solver statistics and per-observable discrepancies. A finite prediction
mismatch is `accuracy_failed`; a numerical failure in a candidate/reference
integration is `accuracy_inconclusive`. A step-limit failure in an accuracy integration retains its structured cause and
uses the existing bounded step-budget recovery (doubling up to the configured
ceiling). Accuracy-only recovery keeps the sample count unchanged and reruns all
validation gates. Other inconclusive failures and accuracy disagreements do not
trigger step increases. If no fully validated seed remains, unresolved failures block readiness without changing the selected solver. Otherwise the fit may proceed from validated seeds with explicit warnings, as described below. Coverage, valid-loss,
fidelity and accuracy checks must still pass: inconclusive accuracy is never
labelled passed. Recovery deadlines include these checks.
This is a sampled forward-convergence test,
not a guarantee for the entire parameter space or for gradients/Hessians.

### Accuracy-validated starting points

Coverage still requires ten successful parameter vectors. The tighter-reference
accuracy gate may proceed when at least one selected probe passes every
experiment and its loss/prediction checks, even if other probes are numerically
inconclusive. Inconclusive probes remain explicitly listed in the report; they
are never labelled accurate. Demonstrated prediction/loss mismatches and
interface errors still block readiness. If any refinement/reference probe is
inconclusive because it hits the step ceiling, budget recovery runs before
excluding that probe, even when other probes passed. It doubles the common
`max_steps` up to `solver_recovery_max_steps` (normally 50,000), with the same
sample count and all validation gates rerun. At the configured cap, remaining
step-limited probes may be excluded with explicit warnings if fully validated
seeds remain. Other numerical failures do not trigger step increases. Recovery
timeouts still block readiness; they never bless incomplete validation. If all
probes remain inconclusive, readiness fails.

Validated normalized points are saved in `solver_accuracy.json`, ordered by
ordinary loss, and bound to the final model/configuration and CSV content hashes.
`pfit run` rejects stale seed reports. DE and PSO insert these points into their
initial populations while retaining an exploratory particle. Search continues
across the original bounds; this does not certify every subsequent candidate.
If the population winner fails at refinement tolerances, gradient refinement
permanently defaults to retrying the **same winning parameters** with the
population-search rtol and atol. A fresh problem is compiled with these
constants, and the winner must have a finite valid loss before Adam starts.
There is no substitution of another parameter seed. If both tolerance profiles
fail, fitting stops explicitly. Gradient-only restarts have no preceding
population stage and retain their requested tolerances.

The run prints a warning whenever it uses population tolerances for refinement:
the requested tighter accuracy was not achieved, but the search winner is
preserved. `fit_summary.json` records `refinement_tolerance_fallback`, requested
and effective refinement tolerances, and the warning. Effective tolerances also
apply to final losses, predictions and sloppiness. The run records its seed
evidence and partial-validation warnings in `accuracy_seeds.json`. Sampled
forward checks do not guarantee derivative accuracy or final fit quality.

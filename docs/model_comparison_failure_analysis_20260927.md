# Failure analysis and proposed scaffolding corrections

The clean benchmark remains unchanged. This investigation uses isolated copies
under `evaluation_runs/failure_analysis_20260927/`; no production framework fix
or replacement benchmark score has been applied.

## Oregonator: confirmed shared scaffolding failure

All three original JAX translations failed the same midpoint integration because
the renderer supplied `max_steps=10000`. The normalized midpoint corresponds to
`eps1=0.04`, `eps2=0.0004`, `q=0.0008`, `f=1.0`.

Direct inspection of Diffrax's result and statistics showed identical outcomes
for all three final generated RHS implementations:

- Maximum solver steps reached; 10,000 total steps.
- 4,988 accepted and 5,012 rejected steps.
- Only 178 of 300 requested output rows finite.
- `_compute_loss_problem` replaced the numerical failure with the generic
  `1e30` sentinel, hiding the actual reason from the smoke-test error.

The unmodified 32B RHS, initial conditions and tolerances completed at the same
point when the step budget was raised to 50,000: 17,802 actual steps (8,866
accepted, 8,936 rejected), all 300 output rows finite, loss approximately
0.9818560937. This is a solver work limit, not a token limit or evidence that the
ODE translation was incorrect.

The decisive verification recovered each model's **original pre-repair fragment
response** from the first repair request, rendered it in an isolated session
with only `gradient_opt.max_steps=50000`, and ran the existing smoke and
source/JAX fidelity gates. **All three passed.** No LLM call, equation edit,
loss edit or tolerance relaxation was needed. This establishes recovery of the
JAX acceptance gate; a full fitting run with this change was not performed.

## The repair loop made Oregonator worse

The workflow treats every smoke-test failure as a reason to request corrected
JAX code. It cannot increase the scaffold-owned solver budget through a fragment
response, so all five attempts per model were ineffective against the cause.

Moreover, comparing the final repaired scripts after increasing the budget
revealed new scientific mismatches:

- 32B's final script still passed source/JAX fidelity.
- 14B's final loss used the wrong measurement columns, including a standard
  deviation as an observed value. Its loss differed from the source by about
  14.08052 at the midpoint. Earlier repair attempts also accessed out-of-range
  `dataset[:, 4]` despite time already being removed.
- Coder-Next changed pooled RMSE into the mean of the two channel RMSEs. Its
  midpoint loss differed from the source by about 0.0001681442, enough to fail
  the existing fidelity gate.

These loss bodies were correct before repair. The invariant should be enforced
in code: a numerical integration failure cannot authorize replacing loss,
writeout, RHS or scientific declarations. A prompt asking the model to preserve
unaffected fields was insufficient.

## Recommended numerical recovery design

1. Preserve structured solver diagnostics: experiment, normalized and physical
   parameters, solver, result code, attempted/accepted/rejected steps, requested
   time interval and finite-output status. Keep the optimizer's scalar sentinel
   but do not use it as the only diagnostic available to validation.
2. Classify failures before selecting a recovery path. A solver step-limit
   failure goes to numerical recovery; an actual source/JAX discrepancy goes to
   translation repair; malformed JSON goes to response-format recovery.
3. For a scaffold-generated/default budget, retry with a bounded step allowance
   (for example 10k -> 20k -> 50k), with a wall-time ceiling. Preserve equations,
   tolerances, data, initial conditions and scientific loss. Respect an explicit
   user hard budget; report exhaustion rather than silently exceeding it.
4. Persist an accepted solver-setting change to the session configuration,
   regenerate the scaffold, and refresh source/config provenance normally so
   validation and fitting use the same setting. Never bypass stale-source checks.
5. Retain the existing source/JAX comparisons after successful integration. Also
   compare RHS/loss/writeout on suitable synthetic finite arrays before an ODE
   solve so scientific translation errors are not masked by numerical failure.
6. A single midpoint is not a universal feasibility test. If a solve fails for
   another classified reason, a small documented set of deterministic parameter
   probes can distinguish an unsuitable point from a broken model. Record failed
   probes and require consistent multi-experiment semantics; do not simply skip
   difficult records or accept the sentinel.
7. Repairs may change only the fields associated with their error category.
   Preserve deterministic source-derived loss/writeout bodies, reject unrelated
   edits, detect repeated identical failed proposals and route back to the stage
   that owns the missing declaration instead of spending five ineffective calls.

## Cascaded tanks: confirmed gradient evaluation problem

The 32B and Coder-Next scripts contain `sqrt(maximum(x, 0))`. At the 32B run's
selected seed, the forward solve succeeded and loss was 1.8033226240240372, but
the gradient had NaNs in two parameter components. Positive saved states do not
prove every internal solver/adjoint evaluation avoids the clipped region.

An isolated replacement evaluated the same piecewise function as:

```python
positive = x > 0
jnp.where(positive, jnp.sqrt(jnp.where(positive, x, 1.0)), 0.0)
```

This keeps the inactive square-root branch away from its singular derivative.
The loss and every saved trajectory value were identical in the probe (maximum
absolute trajectory difference zero), and all four gradient components became
finite: approximately `[-14.30284, 14.30127, -9.69887, 9.54047]`.

This is evidence for a reusable branch-safe translation pattern and a gradient
check at the actual refinement seed. The derivative at the clipping boundary is
not mathematically unique/finite; the zero-gradient convention there must be
explicit. Do not invent a smoothing constant or silently change the supplied
forward model. This probe does not establish full optimizer convergence.

## Other observed failures and scoped corrections

| Failure | Evidence | Scaffolding correction |
|---|---|---|
| NF-kB, all models | Missing fixed `cytoplasm`, explicitly given in prose | Reconcile declarations with the original specification; re-extract only the missing parameter definition and retain its provenance. Never invent a value. |
| Sneyd, 32B | Incorrect missing-input claim for zero initial states; per-record clamp extraction also incomplete | Recheck missing claims against original prose and clamp table; validate state/experiment initial-condition coverage before freezing extraction. |
| Sneyd, Coder-Next | Semantic check returned invalid JSON | Use per-stage structured output schemas and a bounded formatting retry, followed by ordinary semantic validation. |
| Tanks, 14B | Repeated `max(x,0)` in an expression that permits only NumPy calls | Canonicalize supported scalar math AST forms such as two-argument max/min into the appropriate NumPy operations; do not repeatedly ask the LLM to repair spelling. |
| Sliding basepoint, 14B, both variants | Python `and/or` remained in array-compatible expressions | Represent supported predicates/conditionals explicitly and render NumPy/JAX forms, preserving the original piecewise science; do not introduce arbitrary smoothing. |
| Piezo, 14B | Auxiliary output column duplicates an existing model name | Separate output-column references from symbol declarations; reuse an existing validated observable/state rather than redeclaring it. |
| ARC, Coder-Next | Returned `inputs/<file>` when intake expected `<file>` | Normalize the optional `inputs/` prefix for paths resolving inside the input directory; reject traversal, ambiguity and missing files. |

The Ollama client currently requests free-form text and relies on prompt wording
for JSON. Ollama supports a JSON schema in the API `format` field; this should
be used with the stage's response schema, while retaining semantic validation.
See [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs).

## Priority and evidence

First implement classified solver diagnostics, bounded numerical recovery and
field-scoped repairs. This addresses the confirmed shared Oregonator blocker
and prevents damage to valid translations. Next handle branch-safe gradients,
then structured output and targeted extraction/canonicalization failures.
These are proposals; production behavior has not been changed in this analysis.

Reproducible probes and outputs, retained on the persistent volume:

- `evaluation_runs/failure_analysis_20260927/probe_oregonator.py`
- `evaluation_runs/failure_analysis_20260927/oregonator_probe.json`
- `evaluation_runs/failure_analysis_20260927/verify_fixes.py`
- `evaluation_runs/failure_analysis_20260927/fix_verification.json`
- `evaluation_runs/failure_analysis_20260927/verify_original.py`
- `evaluation_runs/failure_analysis_20260927/original_fragments_50000.json`

Run probes with the project venv, `JAX_PLATFORMS=cpu JAX_ENABLE_X64=true`.
The environment prints an optional CUDA plugin/cuSPARSE warning during discovery;
CPU calculations completed. It was not the benchmark's Oregonator failure.

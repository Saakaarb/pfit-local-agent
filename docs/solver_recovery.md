# Solver diagnostics, bounded recovery, and JAX repair restrictions

Implemented after the [three-model failure analysis](model_comparison_failure_analysis_20260927.md).
This change addresses numerical JAX validation and repair routing. Extraction,
check-stage repair, clipped-square-root gradients and model selection are
unchanged. Frozen benchmark scores have not been revised.

## Behavior

Generated scripts expose an additional integration-with-statistics function;
the existing three-value integration interface is preserved. JAX validation
records the actual solver result, attempted/accepted/rejected steps, experiment,
physical and normalized parameters, requested interval and finite output count.
Failed optimization evaluations retain the ordinary scalar loss sentinel.

Only step-limit failures enter numerical recovery. Other solver failures stop
with diagnostics and cannot trigger LLM changes to scientific fragments. New
configurations start at 10,000 steps and explicitly permit recovery up to 50,000,
doubling until capped. Additional attempts run in an isolated process with a
shared 120-second wall deadline including compilation and fidelity checks.
Equations, loss, tolerances and initial conditions remain unchanged.

Existing/custom YAML without `solver_recovery_max_steps` keeps its current
`max_steps` as a hard ceiling. See the [configuration contract](../INPUT_REQUIREMENTS.md#solver-diagnostics-and-bounded-jax-validation-recovery)
for opt-in settings. The wall deadline applies to additional recovery attempts;
the initial ordinary validation remains bounded by its solver step count.

All experiments are revalidated. Accepted step limits persist to YAML, and the
script is regenerated with those settings. Normal fidelity and source-stamp
gates remain mandatory. Exhaustion restores the original YAML and leaves the
script pending. Concurrent source edits cannot acquire a valid stamp through
recovery. Fresh source loading prevents stale bytecode after equal-length
solver-budget edits within one filesystem timestamp tick.

LLM repairs may change only the fragment field identified by the validation
error. Deterministic source-derived loss/writeout bodies are protected.
Unrelated edits are rejected; repeated identical failed proposals stop early.
Unknown failure ownership stops rather than allowing an unrestricted rewrite.
A wholly malformed initial fragment response can still be repaired because
there is no accepted fragment baseline yet.

Artifacts: `generated/solver_diagnostics.json`, `generated/solver_recovery.json`,
worker log/result, and existing workflow events and translation-fidelity reports.
The check-stage repair loop remains shelved.

## Verification

- Full default suite: **324 passed, 8 deselected** before the final bytecode-cache
  regression addition. Subsequent targeted tests: **16 passed**, including that
  new regression and all recovery tests.
- A real two-record Oregonator regression verifies recovery of the long record,
  revalidation of both experiments, source/JAX fidelity, source preservation,
  persisted settings and a valid stamp without LLM repair calls.
- Tests cover absent permission to increase the budget, exhausted caps, worker
  timeout, non-step-limit failures, concurrent source edits, unrelated repair
  rejection, repeated proposals and invalid recovery settings.
- Live Ollama `qwen2.5-coder:32b` translation of the benchmark's Oregonator source
  requested only helpers and RHS. It recorded 10,000-step exhaustion, recovered
  at 20,000, passed fidelity and stamped the recovered configuration.
- The subsequent bounded fit completed **five Adam iterations** without a
  refinement error; loss was **14.744774463898 -> 14.743391562732338**. Post-fit
  sloppiness completed. This verifies execution, not convergence: population
  search used four candidates and one iteration.

Live artifacts: `evaluation_runs/solver_recovery_live_20260927/`. Original
comparison sessions were not modified.

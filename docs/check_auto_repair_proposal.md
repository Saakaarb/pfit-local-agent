# Proposed check-time auto-repair

Status: proposal, not implemented.

Reference: pfit-claude `origin/deployed_branch` at
`1b415d89e7807452126c051f0026a09d14e78504`, `.claude/commands/pfit-check.md`
and `lib/LLM/reference/correction_rules.md`. The reference checks datasets,
corrects YAML/model files, and revalidates for up to three iterations. Local
implementation should preserve deterministic acceptance and user scientific intent.

## User workflow

Normal `pfit check SESSION` would validate, repair eligible critical errors using
configured Ollama/Qwen, and revalidate, for at most three repair attempts.
Stop on success, repeated/no-progress output, or an error requiring a user decision.
Retain `--deterministic-only` and `--ready` as read-only checks. Add `--no-repair`
for semantic review without edits. Do not run fitting or JAX translation during
check; any accepted source edit requires a fresh `pfit jax` before fitting.

## Findings and repair boundaries

Add structured findings (stable rule ID, severity, file/location, measured
evidence, repair eligibility) alongside the current human-readable report.
Deterministic validators establish blocking errors. Unverified semantic findings
remain warnings and cannot authorize edits by themselves.

Eligible examples: syntax/indentation repairs with an unambiguous intended
statement, missing required interface scaffolding whose body is already present,
and inlining a helper expression explicitly supplied in the source specification.
Every proposed edit must cite its finding and supporting source text. A missing
formula or uncertain mapping is unresolved, not permission to invent one.

Never automatically edit CSVs, user_info.txt, historical runs, parameter bounds,
initial conditions, parameter names, observation mappings, loss definitions or
weights, forcing semantics, or optimizer budgets. Do not silently change logscale,
add epsilons, mask observations, or renumber columns to make validation pass.
Sneyd's distinct valid names `k2` and `k_2` require source evidence or a user decision.
Convergence recommendations remain recommendations. These restrictions intentionally
narrow the reference's broad warning/convergence correction policy.

## Implementation

1. Keep the current check prompt responsible for review. Add a separate repair
   prompt that receives only eligible findings, current YAML/Python, relevant
   user specification, and measured column/experiment mappings. Request structured
   exact-text edits with expected old text, replacement, rule ID, and rationale.
2. Restrict patches to `inputs/user_input.yaml` and `generated/user_model.py`.
   Reject unexpected paths, stale matches, unsupported operations, or changes
   outside the eligible finding. Parse YAML and Python before evaluating candidates.
3. Stage each candidate in isolation; run all deterministic session checks across
   every experiment. Reject candidates that introduce new failures or alter
   protected settings or scientific expressions. Preserve the last accepted files
   if a candidate fails; do not accept a patch solely because the LLM approves it.
4. For successful source edits, invalidate translation readiness through the
   existing source-hash mechanism. Do not create a new JAX acceptance stamp.
5. Save original files, candidate diffs, model/configuration, validation reports,
   and rejection reasons under `generated/agent_logs/check_repairs/`. Record
   applied fixes and unresolved decisions in `user_input_check.txt`.

Use the existing workflow's local client, token budget and logging conventions;
cap check repair attempts at three. This needs workflow/validator code as well
as prompt changes: prompt instructions alone cannot enforce acceptance or rollback.

## Acceptance tests

- A repairable syntax/interface error is fixed, then independently passes checks.
- Invalid JSON, repeated patches, new failures and exhausted retries retain valid
  originals and yield a clear unresolved report.
- Attempts to modify CSVs, losses, parameter identities, bounds or settings are rejected.
- Ambiguous valid parameter names remain unresolved without authoritative evidence.
- Every experiment is revalidated, including forcing coverage and missing-data checks.
- Source edits make existing JAX output stale; unchanged checks do not.
- Read-only modes never mutate session inputs or generated source.
- A live Qwen test demonstrates one successful repair and one correct refusal to
  guess, with the complete patch/revalidation record retained.

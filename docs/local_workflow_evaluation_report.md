# Local Workflow Evaluation Report

Date: 2026-09-20

Model: `qwen2.5-coder:14b`

Scratch sessions: `/private/tmp/pfit-suite.uNlpH2`

Scope: existing complete session files were copied to a temp directory and run through:

```text
pfit check -> pfit jax
```

This report does not claim full `pfit new -> pfit check -> pfit jax` success for the
complex repo examples, because those examples keep their detailed model code in
`generated/user_model.py`. `pfit new` intentionally excludes `generated/` from prompt
context, so a true `pfit new` test needs the rough equations/model description supplied
as user input.

## Summary

| Session | Check | JAX | Final Status |
| --- | --- | --- | --- |
| Robertson | passed | passed smoke test | Pass |
| Oregonator | passed | passed smoke test | Pass |
| Boehm STAT5 | passed | failed translation | Fail |
| Sliding Basepoint | passed with warnings | failed translation | Fail |
| Van der Pol | passed with warnings | passed after repair | Pass |

## Results

### Robertson

Result: passed.

`pfit check` completed with convergence-budget recommendations. `pfit jax` rendered
`generated_script.py`, passed generated-script validation, and passed the smoke test.

### Oregonator

Result: passed.

`pfit check` completed. `pfit jax` rendered `generated_script.py`, passed generated-script
validation, and passed the smoke test.

### Boehm STAT5

Result: failed at `pfit jax`.

Observed debug behavior:

- The model produced plausible RHS fragments but referenced intermediate names such as
  `phos_AA`, `phos_AB`, and `phos_BB` without defining/inlining them.
- The model emitted wrong JSON keys:
  - `loss_function`
  - `writeout_function`
- The required keys were missing:
  - `loss_body`
  - `writeout_body`

Final validator output:

```text
translate_jax_fragments: failed: pfit-jax response is missing loss_body
repair_jax_fragments: attempted: 1
translate_jax_fragments: failed: pfit-jax response is missing loss_body
```

Interpretation: this is a local-model contract-following failure. The validator is doing
the right thing by rejecting the response.

### Sliding Basepoint

Result: failed at `pfit jax`.

`pfit check` passed but warned about dataset/YAML shape mismatch and writeout shape
inconsistency.

Observed debug behavior:

- The model generated helper functions `F1` and `F2`.
- `F2` referenced `vf` but did not take `vf` as an explicit helper argument.
- Repair repeated the same helper dependency error.

Final validator output:

```text
translate_jax_fragments: failed: pfit-jax fragment uses unknown name: vf
repair_jax_fragments: attempted: 1
translate_jax_fragments: failed: pfit-jax fragment uses unknown name: vf
```

Interpretation: this is a helper dependency hygiene failure. The model understands much
of the scientific structure but does not reliably satisfy the fragment contract.

### Van der Pol

Result: passed after repair.

`pfit check` passed with warnings from the semantic reviewer. `pfit jax` initially failed
because the loss body did not return a value, then the repair step fixed it. The rendered
script passed validation and smoke testing.

Final validator output:

```text
translate_jax_fragments: failed: pfit-jax loss_body must return a value
repair_jax_fragments: attempted: 1
render_generated_script: created
validate_generated_script: passed
smoke_test_generated_script: passed
```

## Findings

The current local workflow is working for simpler and moderately complex sessions:
Robertson, Oregonator, and Van der Pol.

The remaining failures are concentrated in complex JAX translation:

- preserving RHS intermediates correctly
- converting intermediates into explicit helper functions or inlining them
- ensuring helper functions take every dependency as an explicit argument
- maintaining the required JSON fragment schema under repair

This is not a time-column indexing issue anymore. The dataset runtime contract fix worked
for the simple case and did not regress the passing examples.

## Next Steps

1. Re-run Boehm and Sliding Basepoint with `qwen2.5-coder:32b`, because the 14B model is
   currently failing on contract fidelity.

2. Consider adding a second repair guard for invalid repair outputs: if repair returns a full
   Python script or Markdown-fenced code instead of JSON, fail immediately with a short message
   rather than allowing a long off-contract generation.

3. If 32B still fails Boehm/Sliding, split `pfit jax` translation into smaller structured
   subtasks:
   - translate RHS and helpers;
   - validate helpers;
   - translate loss;
   - translate writeout.

## Follow-Up Improvement

After this evaluation, the `pfit jax` prompt and validator were strengthened:

- RHS intermediate variables must now be explicitly inlined or translated into helpers.
- Helper functions are explicitly told not to close over fixed parameters, trainable
  parameters, state values, or local intermediates.
- Unknown-name validator errors now include the fragment location and a concrete repair
  instruction.

Focused tests were added for:

- unknown RHS intermediates;
- helper functions with missing explicit dependencies.

Full suite after these changes:

```text
96 passed, 4 deselected
```

Sliding Basepoint was rerun with `qwen2.5-coder:14b` after the improvement. The first
translation still used `vf` inside helper `F2` without passing it explicitly, but the
validator now produced a targeted error:

```text
pfit-jax helper function F2 uses unknown name: vf. Add vf as an explicit argument to F2
and update every call, or inline the value before calling F2.
```

The repair attempt then drifted into a full Markdown-fenced Python script instead of JSON,
so the final failure became:

```text
translate_jax_fragments: failed: pfit-jax returned invalid JSON
```

Interpretation: the framework diagnostics improved, but `qwen2.5-coder:14b` still does
not reliably repair this complex fragment. The next useful model test is `qwen2.5-coder:32b`.

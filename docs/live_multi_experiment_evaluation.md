Live multi-experiment evaluation — 2026-09-26

Final outcome after the fixes below: both decay workflows and reference-seeded
Sneyd passed. Fresh Sneyd passed after user-authorized manual corrections to
Python and translated JAX equations. Autonomous fresh fidelity remains open.
The first table records the original baseline failures, not the final outcome.

Tested commit `c915677` with Ollama 0.34.4 and `qwen2.5-coder:32b`
(Q4_K_M), served on the RunPod H100. The Python client, native executable and
model were already installed; restarting `/workspace/start-ollama.sh` restored
service. The real connectivity pytest passed (1 test, 77.58 seconds).

| Case | Fresh extraction workflow | Reference-seeded workflow |
| --- | --- | --- |
| Decay, two experiments | Passed new/check/jax/run/diagnose | Passed check/jax/run/diagnose |
| Sneyd IPR, nine experiments | Failed new: undefined `v9` after two repair attempts | Failed check: incorrect demand for extra loss normalization |

Both decay paths preserved two experiments and matched reference per-record
losses at the reference parameters and a 2% parameter perturbation (rtol 1e-3,
atol 2e-6). Bounded fits used population size 4, one population iteration and one
L-BFGS iteration. Mean losses fell from about 0.316666 to 0.0208081. These are
workflow smoke tests, not convergence benchmarks. Fitting subprocesses used CPU;
objective comparisons used the default available JAX backend.

Fresh Sneyd extraction retained rate aliases in the differential equations without
providing the required definitions. Deterministic validation rejected `v9`; both
repair responses supplied observables instead of repairing the equations.
The reference-seeded check claimed the user wanted normalized/scaled residuals.
The supplied specification explicitly requests plain RMSE of open probability;
the reference comment calls this normalized because the observable is already a
probability. The rejection was initially described as a semantic-review false positive; the
subsequent trace below identifies the deterministic parser as its source. Neither live
Sneyd path reached JAX translation or fitting. Existing deterministic nine-record
numerical tests remain separate evidence, not a substitute for a live pass.

Fresh sessions were supplied CSVs and an explicit problem description containing
the reference YAML specification and Python equations/loss. Seeded sessions
started with reference YAML and Python model code. This evaluation therefore does
not establish extraction quality from papers or unstructured prose. Temperature
was 0.1, context length 32768, response limit 12000 tokens, request timeout 600
seconds, and repair limit two. No code fixes or bypasses were applied to make
failed cases pass. OPEN-07 remains open.

Reproducible runner, exact inputs, generated files, model digest, stage timings,
per-record comparisons and failure logs are retained under
`evaluation_runs/live_multi_20260926_175202/` in persistent `/workspace`.
See `manifest.json`, `run_live.py`, and `logs/`. These ignored artifacts are local
to this RunPod volume; this document is the repository summary.

Next work: repair missing intermediate expressions during fresh extraction and
resolve the semantic review's normalization false positive, then rerun both
Sneyd paths through all nine experiment outputs.

Failure analysis and workflow improvements

The loss-contract parser recognized `Loss:` alone and headings starting with
`Loss `, but missed `Loss: plain RMSE ...`. Its fallback returned the entire
user_info.txt, including appended reference code. The comment “plain RMSE is
normalized” then matched the normalization keyword heuristic, producing a
critical error for missing division. That deterministic error was passed to the
LLM reviewer, which repeated it. The explicit one-line declaration is now
recognized and stops at the next section heading. The checking prompt also
clarifies that already normalized observables and plain RMSE do not require
additional residual scaling.

Equation extraction returned the Sneyd RHS in terms of v0 through v9 but supplied
only L1/L3/L5 formulas. Validation correctly rejected the unresolved aliases.
The old repair schema allowed RHS/observable edits but no formula definitions;
both replies edited only the observable. Both expression and loss-body repairs
also had a 350-token ceiling. The recorded responses were short, so truncation
has not been established as the cause; the ceiling nevertheless constrained
complete repairs of larger systems.

Extraction instructions now explicitly require all intermediate rate dependencies.
Expression repairs may return formulas, which the existing formula-inlining code
expands into RHS and observable expressions before normal validation. No new
state or parameter declarations are accepted through this repair route. Both
repair ceilings are now 4096 tokens, bounded by the user's configured max_tokens.
Scientific definitions must still come from the supplied context, not guesses.

Focused regression coverage checks one-line loss section boundaries, the actual
Sneyd reference loss, chained missing-rate repair, and both default and smaller
configured response budgets. Live rerun artifacts are kept separately under
`evaluation_runs/live_multi_sneyd_improved_20260926/` to retain the original
failures unchanged.

The first improved live rerun passed fresh equation extraction but exposed a
separate objective-rendering defect. The loss response explicitly contained
`metric: "rmse"` with `custom_loss: false`; the renderer discarded all supplied
terms when that flag was false and emitted default MSE. Deterministic checking
correctly rejected the missing square root. Supplied terms/penalties now take
precedence over that flag, plain `rmse` has an explicit renderer branch, and the
loss-extraction prompt treats any explicit metric as a supplied loss. A numerical
regression distinguishes RMSE from MSE and additional scaling. The subsequent
clean fresh run is in `evaluation_runs/live_sneyd_fresh_rmse_20260926/`.

Improved reference-seeded Sneyd passed check, live JAX translation (two repairs),
all-nine-record smoke validation, numerical loss comparison at both parameter
vectors, bounded fitting and diagnosis. Nine result CSVs were verified. The mean
loss decreased from 0.08579784115437102 to 0.0840942304090551 with one L-BFGS
iteration after the small population search. This demonstrates the live seeded
workflow, not convergence to the saved reference fit. Both decay paths also
passed the first improved evaluation.

After the RMSE fix, fresh Sneyd passed new/check/jax but failed the independent
reference objective comparison on all nine records (maximum absolute difference
0.06322362 at the first parameter vector). Extraction changed `k_2` to `k2` in
L3 and `k_4` to `k4` in L5. The saved equation-extraction prompt contained the
correct source assignments, so this was a model transcription error, not missing
input. Syntax/smoke checks cannot establish scientific equivalence. The evaluation
stopped before fitting. A final instruction asks for exact copying of scalar
assignments, distinguishing identifiers with underscores and leaving expansion to
the framework; its clean live trial is under
`evaluation_runs/live_sneyd_formula_copy_20260926/`.

Default regression suite after the implementation changes: 270 passed,
7 deselected. No previously failed evaluation artifacts were overwritten.

The final exact-copy-instruction trial also passed new/check/jax but failed the
reference objective comparison; it reproduced the same L3/L5 parameter-name
changes. The saved request confirms the new instructions reached the model.
No fit was attempted on either fresh model after numerical fidelity failed.
Prompt wording alone has not resolved this transcription issue. A next improvement
should verify extracted equations against supplied source equations before
acceptance, with targeted repair for discrepancies; the current generic readiness
checks establish validity/executability, not scientific equivalence. Reference
objective comparisons here belong to the evaluation harness and are not yet a
general production gate.

Current outcome: both live decay workflows pass, reference-seeded nine-record
Sneyd passes through fitting/diagnosis, fresh Sneyd remains blocked by reference
fidelity. Implementation changes and these notes accompany the workflow-fix commit.

User-assisted continuation

At the user's request, a copy of the final fresh session was manually corrected
in `evaluation_runs/live_sneyd_manual_correction_20260926/manual/sneyd_ipr/`.
The correction restores L3 = k_2*l4/(k2*l_4) and L5 = k_4*l6/(k4*l_6)
throughout the expanded RHS (8 and 10 occurrences, respectively). The exact diff
is retained as `manual_correction.patch`. Original failed sessions are preserved.
The corrected Python RHS matched the reference exactly at 25 sampled states and
parameter vectors (seed 17); verification code and results are retained beside
the session. The continuation starts at check and uses live Ollama JAX generation;
it does not rerun new or copy reference generated JAX code. This is explicitly a
user-assisted result, not a successful autonomous fresh extraction.

Live translation of the corrected Python then introduced a further transcription
error in dI2dt: `k_1*l2/(k1*l_2)` became `k_1*l2/(k_1*l_2)` in a denominator.
The independent reference comparison rejected it. Under the same user-authorized
manual-correction workflow, this one generated JAX expression was restored.
All eight JAX RHS expression ASTs were then verified equal to the corrected Python
RHS. The pre-correction generated script, exact patch and verification are saved.
The initial comparison remains in manifest.json; subsequent fidelity/fit results
are recorded separately in continuation_manifest.json. No autonomous translation
fidelity success is claimed for this manually repaired session.

Manually corrected Sneyd completed the reference objective comparison (all nine
records at both parameter vectors), bounded fitting and diagnosis. Mean loss
decreased from 0.0857978411544 to 0.0840942304091.
All nine output CSVs contain finite values, and their reported record losses
average to the final loss. The run used population size 4, one population
iteration and one L-BFGS iteration; this is a workflow validation, not a converged
fit. Run ID: `run_20260926_183937_720530`. Results and corrections are preserved under
`evaluation_runs/live_sneyd_manual_correction_20260926/`.

Multi-experiment support plan

Status: implemented in the working tree within the first-release contract. See
[multi_experiment_support.md](multi_experiment_support.md) for evidence and limits.
Original plan written 2026-09-26 against local commit
`386e061` and `pfit-claude origin/deployed_branch` at `1b415d8`.
Tracks OPEN-01 in [feature_parity_status.md](feature_parity_status.md).

Goal and reference behavior

Fit one shared parameter vector to all experiment records, integrating each
record independently with its own CSV, time grid and initial conditions. Preserve
`pfit new -> check -> jax -> run -> diagnose`, including gradient-only restart
and snapshot-based sloppiness analysis.

Port the deployed design rather than inventing another experiment framework:

- `lib/utils/yamlread.py`: `experiments` records and `get_y0(index)`, merging
  per-record `initial_conditions` over global model initial values.
- `lib/utils/helper_functions.py`: one constants dictionary per experiment,
  one shared per-experiment generated function, and an arithmetic mean of losses.
- `tests/fixtures/decay_multiexp`: two analytic decay experiments with different
  initial conditions and shared true parameters `k1=1.0`, `k2=0.3`.
- `sessions/sneyd_ipr`: nine CSVs; calcium/IP3 clamp conditions represented by
  per-experiment initial values of zero-derivative states. Its objective is the
  mean of per-experiment open-probability RMSEs.

First-release contract

1. Keep the deployed YAML shape: each experiment has `data_file`, `columns`,
   and optional `initial_conditions`. No new required IDs or weighting fields.
   Use YAML order as the stable experiment index and record its source filename.
2. All experiments share equations, trainable/fixed parameters, parameter bounds,
   and solver settings. Only time grids, measurements and initial states vary.
   Different row counts and irregular, strictly increasing time grids are valid.
3. All records must use the same ordered column meanings: names, observation
   mappings and uncertainty roles. Validate meanings, not just equal widths.
   Reject mismatched layouts rather than silently remapping positional losses.
4. Initial-condition overrides must name integrated states and have finite values.
   Omitted overrides inherit global values. Without explicit `initial_time`, each
   experiment starts at its own first observation. An explicit global initial time
   must precede or equal the first observation of every record.
5. The objective is `L(theta) = mean(L_1(theta), ..., L_E(theta))`, with equal
   experiment weight. Preserve each existing per-record user loss exactly:
   mean-of-RMSEs is not pooled RMSE, and equal record weight is not row weighting.
   Record this aggregation rule in run metadata and reports.
6. A failed/nonfinite per-record evaluation invalidates the whole candidate.
   Preserve the full error-loss sentinel instead of averaging it down. Population
   search, refinement, restart seed evaluation and sloppiness must use the same
   aggregation and failure policy.
7. Preserve single-experiment objectives, CLI commands, restart behavior and
   `result_solution.csv`. For multiple records use the reference convention
   `result_solution_exp1.csv`, `result_solution_exp2.csv`, etc., plus a manifest
   mapping each output to its experiment. Other writeout labels follow the same
   rule. This single-record filename compatibility is an intentional local choice.

Do not bundle measured forcing, experiment-specific fitted parameters, arbitrary
column remapping, custom experiment weights, Adam exposure or a broad scientific
loss redesign into this port. Existing missing-measurement/uncertainty behavior
must not regress; this work does not close PART-03. Custom losses remain subject
to per-record smoke checks rather than being silently replaced with another loss.

Implementation sequence

1. Reader, experiment loading and deterministic validation

   Adapt `lib/utils/yamlread.py` to retain every record and expose the deployed
   `get_y0(index)` behavior. Retain first-record compatibility accessors only for
   existing single-record callers; migrate all runtime consumers explicitly.
   Introduce one small shared experiment-loading helper under `lib/utils/`, using
   the existing CSV loader, for fitting, smoke tests and historical reanalysis.

   Extend `SessionValidation` and `SessionSpec` to describe every experiment.
   Run current data/time/width checks on each record; add IC-override and common
   column-schema checks. Errors identify experiment index and filename. Check
   uncertainty declarations against each record's declared measurement columns.
   Missing measurements must not cause dropped rows or dropped experiments.

   Keep the public multi-experiment rejection in place until dependent runtime
   paths are complete. Reader/loader tests can exercise multiple records without
   advertising incomplete CLI support; move the temporary gate to the workflow
   boundary if needed during development.

2. Shared fitting objective and per-experiment output

   Adapt `CreatedClass` in `lib/utils/helper_functions.py` to own an experiment
   list and `constants_list`. Keep the generated function signatures unchanged:
   each call still receives one record's constants and the shared parameter
   vector. Use the reference's trace-time Python loop, allowing unequal lengths
   without padding or concatenating time grids.

   Implement the aggregate objective once and reuse its failure semantics in
   sloppiness. Update every bounds/logscale/tolerance setter across all constants.
   Build separate population and gradient problem objects, preserving loose/tight
   tolerances for every record. Migrate full and gradient-only fitting together.
   The optimizer still receives one scalar objective and one shared vector.

   Write one trajectory per record and save per-record final losses alongside
   the aggregate. Preserve custom writeout columns. Update result-file consumers
   to use the experiment manifest rather than assume one CSV.

3. Complete run snapshots, restarts and sloppiness

   Update `local_agent/core/fitting.py` to copy every dataset to distinct snapshot
   paths (`dataset_1.csv`, etc.) and rewrite every runtime YAML record. Preserve
   IC overrides, original filenames/order, column declarations and aggregation
   metadata in the snapshot/manifest. Fit only from the complete snapshot.

   Update `analyze_fit.py` to rebuild all experiment constants from that snapshot;
   feed the complete list to the existing sloppiness API. Remove remaining
   single-record objective assumptions. Test that the Hessian is that of the
   same averaged loss used by optimization, in the existing parameter coordinates.

   A gradient-only restart remains a warm start of the current session objective:
   validate shared parameter compatibility, reevaluate the seed against every
   current record, create a fresh snapshot, and preserve the original run. If
   experiment data/ICs changed, metadata must make that visible; historical
   reanalysis always uses the old snapshot instead.

4. Local-agent extraction, translation, checking and diagnosis

   Update the dataset-selection/spec pipeline in `session_init.py` and its current
   prompts to enumerate relevant supplied CSVs and explicit per-record ICs.
   Preserve the existing single-dataset response format as a compatibility input.
   Never assume every CSV is another experiment: distinguish experiment data from
   auxiliary files, and report missing experiment context when it cannot be
   determined. Do not infer unprovided initial conditions from measured outputs.

   Extend prompt context and `SessionSpec` summaries with every record, keeping
   shared equations and the common observation mapping explicit. Generate one
   reusable RHS/loss/writeout contract through the existing LLM translation step.
   State that functions process one experiment and the framework averages losses;
   this prevents the model from aggregating the experiments a second time.

   Run deterministic generated-code smoke tests for every experiment, checking
   finite successful loss and writeout dimensions on that record's own time grid.
   Stamp accepted translations only after all checks pass. Extend offline checks,
   semantic-review context, and diagnosis to account for every record and output.
   A failed later record must block acceptance even when record 1 passes.

5. End-to-end fixtures, Sneyd verification and release

   Port the small deployed decay fixture first, using the local DE/L-BFGS path.
   Keep the reference's equations, ICs and data; record any local optimizer-setting
   differences. Add an unequal-length/time-grid variant to exercise shape handling.
   Use it for fast deterministic, full-fit, restart and reanalysis regressions.

   Then port Sneyd's nine-record configuration/data/model and translate through the
   local workflow. Verify all nine clamp conditions, independent trajectories,
   derived open-probability outputs, per-record RMSEs and their arithmetic mean.
   Separate deterministic reference-code tests from live Ollama translation tests;
   record the model and repair attempts if live translation is evaluated.

   Check numerical integration at a known feasible reference parameter point,
   then run a bounded fitting smoke test. Do not make successful identification
   of all Sneyd parameters a prerequisite or infer scientific recovery from a
   short smoke fit. Any long fit/evaluation should have explicit budget and
   reported convergence evidence.

   Remove the public multi-experiment rejection only when the complete workflow,
   snapshots, restart and diagnostics pass. Update INPUT_REQUIREMENTS.md, README,
   the parity tracker, and paper-alignment notes with the implemented scope.

Acceptance tests that protect against silent first-record handling

| Test | Required outcome |
|---|---|
| Reader with two records and partial IC overrides | Both retained; inherited and overridden states match expected vectors. |
| Bad time grid, missing CSV or unknown IC key only in record 2 | Check fails and names record 2. |
| Same width but reordered/different observable meanings | Check fails before fitting. |
| Different row counts and time grids | Independent trajectories retain their original lengths/times. |
| Analytic per-record losses | Aggregate and gradient equal the arithmetic means; unequal row counts do not change record weights. |
| Change only record 2 in a controlled identifiable fixture | Aggregate loss, gradient and fitted optimum change as analytically expected. |
| Integration failure only in record 2 | Whole candidate fails; sentinel is not diluted and invalid gradients do not update the fit. |
| Existing NaN/uncertainty case in a later record | Valid missing observations are preserved and handled by its loss; invalid simulation is not masked as missing data. |
| Full fit followed by gradient-only restart | Every record contributes in both stages; all outputs are present; prior run remains untouched. |
| Snapshot isolation | Editing/deleting working datasets cannot alter saved-run reanalysis; all original record mappings remain recoverable. |
| Sloppiness objective | Analytic/finite-difference Hessian agrees with the fitted aggregate, not record 1 alone. |
| Fake-LLM new/check/jax path | All supplied experiment declarations survive; a failed later-record smoke test prevents a valid source stamp. |
| Single-experiment regressions | Existing objective values, filenames and CLI behavior remain compatible. |
| Sneyd IPR | Exactly nine independent records, correct IP3/Ca overrides, nine outputs and verified mean RMSE. |

Suggested review units

A. Experiment schema/loader and validation, with the runtime feature still gated.
B. Multi-record engine, snapshots, restart, outputs and sloppiness as one usable core.
C. Local-agent generation/checking/diagnosis, fixtures and documentation; enable
   public support after the end-to-end acceptance checks pass.

The key release criterion is not merely loading nine files. Every declared record
must affect the objective and appear in validation, saved provenance and results.

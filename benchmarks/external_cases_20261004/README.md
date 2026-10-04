# New external problem set — 2026-10-04

Six published case variants have been assembled. They are separate from the
September 27 and October 3 model comparisons. The initial 32B smoke run passed
Beer and Raia and failed Schwen during extraction. The completed comparison retained those
three attempts, ran the three additions with 32B, and ran all six with 14B.
32B passed 3/6; 14B passed 1/6. All failures occurred during extraction, before fitting.
Completed results: [external comparison](../../evaluation_runs/external_comparison_20261004/comparison.md).
The machine-readable list is [inventory.json](inventory.json); add future run
identifiers there rather than changing the historical comparison tables.

| Case | Scientific use | States / fitted parameters | Records / measurements | Status |
|---|---|---|---|---|
| [Beer](cases/beer_indigoidine/inputs/user_info.txt) | Engineered indigoidine production coupled to bacterial growth | 4 / 5 | 1 / 1,428 | 32B passed; 14B passed |
| [Raia](cases/raia_il13/inputs/user_info.txt) | IL-13/JAK/STAT signalling and feedback in lymphoma | 14 biological + 1 ligand clamp / 15 | 2 / 33 | 32B passed; 14B extraction failed |
| [Schwen](cases/schwen_insulin/inputs/user_info.txt) | Insulin uptake, binding and receptor trafficking in hepatocytes | 11 / 15 | 4 / 64 | Both extraction failed |
| [Armistead](cases/armistead_sphingolipid/inputs/user_info.txt) | Sphingolipid metabolism in wild type and Hai1a deletion | 4 biological + 1 condition clamp / 10 | 8 / 48 | Both extraction failed |
| [Borghans](cases/borghans_calcium/inputs/user_info.txt) | Calcium-store and IP3 feedback oscillator | 3 / 19 | 1 / 111 | 32B passed; 14B extraction failed |
| [Fujita](cases/fujita_egf/inputs/user_info.txt) | EGF-induced EGFR–Akt–S6 signalling | 9 biological + 1 ligand clamp / 16 | 2 / 48 | Both extraction failed |

These are not toy decay or predator–prey exercises. They test coupled kinetics,
hidden states, nonlinear feedback or assay response, multiple experimental
conditions, and potentially weak parameter identifiability. Beer is the smallest
but includes nonlinear product formation and two dense measured trajectories.
The counts above describe our prepared variants, not the full source studies.

## What a user supplies

Each `cases/<name>/inputs/` contains only a scientific `user_info.txt` and CSVs.
The framework requires the name `user_info.txt` (the requested user-input text
file). Prompts use prose, derivative notation, named reaction rates and ordinary
mathematical products. There is no YAML, executable model, JAX code, schema
instruction or reference solution in the input directories. These are newly
authored prompts intended to be plausible for a scientist to write; they are
not claimed to be independently collected human submissions.

For a future run, copy **only** `cases/<name>/inputs/` to a fresh session's
`inputs/`. Do not give the entire cohort directory to pfit: `sources/` and the
inventory are provenance evidence, not LLM inputs. The runner copies only these input folders; source material is never part of the LLM input.

Schwen and Borghans explicitly give initial conditions at t=0 while observations start at
0.25 and approximately 0.0295 minutes, respectively. The integrator supports `gradient_opt.initial_time: 0`, but current
new-session rendering does not emit it from prose. Before their check/JAX
steps, ensure that setting is present in the generated YAML. It is recorded in
`inventory.json` as required setup, not silently treated as supported extraction.
Do not shift the observation times or fabricate a zero-time measurement.

## Scientific scope and changes

These are **adapted fitting tasks**, not exact reproductions of the original
PEtab objectives. The full upstream model, conditions, parameter table,
measurement table and observable definitions are retained for auditing.

- Beer uses one T1 experiment, retains every time and both channels, and fixes
  the lag and initial bacterial density at archived nominal calibration values.
  Five kinetic/growth parameters remain free. The loss is normalized pooled RMSE.
- Raia retains all pSTAT5 observations at two ligand doses and the full biological
  state network. It fixes the internal-receptor initial value and the two rates
  on branches that do not influence pSTAT5. The 15 remaining rates are shared.
  The loss is the mean of two per-record normalized RMSEs.
- Schwen retains two doses from one experimental series, including both
  observations at each time. Duplicate observations are distributed by within-time
  order into two records per dose; these are bookkeeping records, not asserted
  biological replicate trajectories. Initial receptor pools are fixed at nominal
  calibration values. Eleven kinetic and four assay parameters remain free.
  The loss is the mean squared log10 residual, without the upstream priors or
  fitted noise parameter. Equal-size records preserve equal per-point weighting.

- Armistead retains both conditions and all repeated measurements for Sphinga,
  Cer and Sphingo. S1P baseline-only assays are omitted; its dynamical state
  remains. Eight bookkeeping records preserve the measurements without claiming
  longitudinal replicate identity. Ten rates/modifiers are shared.
- Borghans retains the complete three-state nonlinear oscillator and all 111 Ca
  measurements. Initial states use fixed nominal calibration values; 19 kinetic
  and assay parameters remain free. The objective is normalized signal-space
  RMSE, an explicit change from the source log10 observation/noise model.
- Fujita retains all three channels at all eight times for EGF steps of 1 and
  10 ng/mL. Nine biological states and all 13 rates are retained, plus three
  fitted observation scales. Initial pools use fixed nominal values. The source
  receptor synthesis constant 68190 remains unchanged. The clamped ligand
  represents the step exactly over the retained observation window.

All retained measurement values and times are copied without interpolation,
averaging, imputation or synthetic replacement. `data_provenance.json` maps each
output row to the original TSV line(s). Fixing initial values at existing fitted
values uses prior calibration information; these variants are not blind recovery
tests. Sparse observation of a large network can yield non-identifiable rates;
that is a useful diagnosis challenge, not evidence of a unique estimate.

Original search bounds are retained for the selected parameters. Preparation
does not establish that random parameter draws integrate successfully or that
optimization converges. No model output has been used to tune these new prompts.

## Sources and reproduction

Source collection: [Benchmark-Models-PEtab](https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab),
pinned to `fcbddf1b900efabdfbdc2b58452c89556e63f1ce` on 2026-10-04.
Exact paths and licensing are in [sources/upstream.json](sources/upstream.json)
and [sources/LICENSE](sources/LICENSE). Source XML files are archived evidence
only; this adds no XML input path or importer to the framework.

- [Beer et al., Molecular BioSystems (2014)](https://doi.org/10.1039/c3mb70594c):
  engineered IndC variants and indigoidine production.
- [Raia et al., Cancer Research (2011)](https://doi.org/10.1158/0008-5472.CAN-10-2987):
  modelling IL-13-induced signalling in lymphoma to study therapeutic targets.
- [Schwen et al., PLOS ONE (2015)](https://doi.org/10.1371/journal.pone.0133653):
  hepatic pharmacokinetics, including insulin uptake. The upstream case identifier
  `Schwen_PONE2014` is retained for traceability; the publication is from 2015.

- [Armistead et al., Cell Death & Disease (2024)](https://doi.org/10.1038/s41419-024-07134-2):
  sphingolipid control of apoptosis versus apical cell extrusion.
- [Borghans et al., Biophysical Chemistry (1997)](https://doi.org/10.1016/s0301-4622(97)00010-0):
  mechanisms of complex intracellular calcium oscillations.
- [Fujita et al., Science Signaling (2010)](https://doi.org/10.1126/scisignal.2000810):
  filtering of receptor signals by the Akt pathway.

`python prepare_data.py` reconstructs CSVs from the pinned TSVs only.
`python verify_assembly.py` checks file hashes, finite values, increasing times,
row-level provenance and expected counts. Neither calls pfit, an LLM, an ODE
solver or an optimizer. Handwritten prompts are kept separate from conversion.

Additional screened candidates are recorded under `deferred_candidates` in the
inventory. They are not ready cases: Crauste needs a missing-observation strategy;
Weber needs preequilibration handling; Bruno needs a fuller condition-mapping
adaptation; Elowitz needs initial-condition and reporter-source review.

## Smoke comparison protocol

Run `scripts/run_external_smoke.py --root evaluation_runs/external_comparison_20261004 --models qwen32b qwen14b --reuse-root evaluation_runs/external_smoke_20261004`
with the repository venv and Ollama environment. The root must not exist.
The three imported 32B attempts keep their complete original artifacts and timings;
input hashes, model name, settings and initial-time setup are checked before import.
Only benchmark assembly/runner files change between these attempts and the new batch;
production extraction and fitting code is unchanged. No manual repair is applied
to Schwen during this comparison.

Both models use identical sampling, context and repair limits. Fitting is CPU/float64
with DE population 4 for one iteration, followed by five Adam iterations; sloppiness
is disabled. Logs, per-stage times, LLM token/timing records, model sizes, digests
and parameter counts are retained. Both models are non-thinking Qwen2.5-Coder
Q4_K_M variants. A pass means the workflow completed its checks; it does not prove
that the extracted scientific model is correct or that the parameters converged.

The broad original search bounds are intentionally retained. In particular,
Borghans permits a very large Hill exponent and Fujita spans extreme time scales.
Numerical failures are legitimate benchmark outcomes and are not removed by
retuning the prompts after seeing model output.

Comparison of deployed pfit-claude, the manuscript, and pfit-local-agent

Current implementation status and remaining work: [feature_parity_status.md](feature_parity_status.md).

Reviewed 2026-09-26. Correct reference: `pfit-claude origin/deployed_branch`, commit
`1b415d8`. Paper: `mimb_agentic_ode_overleaf_1_ (2).pdf` (18 pages).

**Correction to the initial review:** I initially inspected the checked-out `main`
branch (`31f46ff`), which contains the older XML/three-command implementation.
The deployed branch contains the five-command YAML workflow, multiple experiments,
gradient-only restart, sloppiness, preserved run snapshots, a live dashboard, and
post-fit diagnosis tooling. The initial statements that these features were absent
from the supplied original repository were incorrect. This document supersedes
that comparison. Local gaps below refer to the local agent before this task's port.

The first implementation work is documented in
[restart_and_sloppiness.md](restart_and_sloppiness.md): adapt the deployed restart
and sloppiness implementations, with the local CLI and single-dataset backend.

**Reference implementations to reuse**

These paths are on `origin/deployed_branch`; use `git show
origin/deployed_branch:<path>` while the reference checkout remains on `main`.

| Capability | Deployed source |
|---|---|
| Gradient-only driver and saved physical seed | `fit_gradient_only.py` |
| Gradient-only numerical workflow | `lib/utils/helper_functions.py::fit_gradient_only_system` |
| Sloppiness and eigenvalue spectrum plot | `lib/utils/sloppiness.py` |
| Analysis of an existing saved fit | `analyze_fit.py` |
| Run selection, snapshots, seed preservation | `lib/utils/run_store.py` |
| Multi-experiment loading and fitting | `lib/utils/dataset_io.py`, `lib/utils/yamlread.py`, `lib/utils/helper_functions.py` |
| Run readiness and gradient-only seed checks | `tools/check_ready.py` |
| Live dashboard and progress | `lib/utils/live_dashboard.py`, `live_progress.py`, `live_view.py` |
| Autodiff diagnosis | `tools/autodiff_diagnose.py`, `lib/LLM/reference/autodiff_diagnosis.md` |
| Agent procedures and scientific guidance | `.claude/commands/`, `lib/LLM/reference/` |
| Existing regression coverage | `tests/test_sloppiness.py`, `test_fit_configurations.py`, `test_run_store.py`, `test_entrypoints.py` |

**Local-agent changes, ordered by importance**

| Priority | Local gap | Required change and manuscript impact |
|---|---|---|
| Selected | Gradient-only restart missing from local CLI/engine | Port the deployed warm-start behavior: load saved physical parameters, bypass population search, preserve the source run, record seed provenance, and run current gradient settings. Section 3.3, p.14. |
| Selected | No local sloppiness analysis | Port the deployed Hessian analysis, AD-to-finite-difference fallback, report and spectrum plot. Retain its size limit and nonfatal diagnostic behavior. Qualify stationarity, boundaries, negative/flat curvature, and numerical failure. Section 2.3, p.6. |
| P0 | Local YAML reader selects only `experiments[0]` | Port experiment records through extraction, schema, initial conditions, fitting, loss aggregation, output and diagnostics. Until then, explicitly reject unsupported multiple-record input. The drug example on pp.10–13 needs this. |
| P0 | Measured forcing is not represented end to end | Local RHS fragment validation excludes `dataset`/`t_eval` despite generated bindings. Add explicit input-column roles, forcing interpolation, coverage and missing-forcing checks. Distinguish analytic time dependence, which is supported. Sections 3.1–3.1.3. |
| P0 | Missing-data guarantees exceed local implementation | Structured losses use ordinary means/extrema without general masks; standard JAX normalization takes a maximum before masking, allowing NaN scale. Port/test safe measurement masks, uncertainty handling and empty-channel policies. Invalid simulation values must not disappear as missing observations. pp.8, 11–13. |
| P0 | Custom loss/writeout can change during local translation | `_standard_loss_and_writeout_bodies()` selects both from the loss classification, without checking for a custom writeout. Preserve each function independently and compare source/JAX values at representative points. p.14. |
| P0 | Loss transformation policy is inconsistent across steps | Checker updates honor explicit user objectives, but `_apply_automatic_log_loss()` in new-session creation can still rewrite certain normalized MSE terms. Apply explicit-loss precedence throughout new/check/jax, document default-only transformations and offset-log semantics. pp.9–13. |
| P0 | Preflight lacks deterministic numerical checks | Validate finite/strictly increasing time, finite ordered bounds, positive logarithmic bounds/tolerances, array dimensions, and feasible numerical settings. Port the deployed validation rules with local schema adaptations. pp.8 and 13. |
| P1 | `pfit check` reports without correcting | Add bounded mechanically justified correction/revalidation, preserving declared equations, constants, data mappings and loss. Local new/jax repairs do not cover check-time correction. Section 3.2. |
| P1 | Local numerical configuration is limited | New sessions hard-code small DE/refinement budgets; NODE hard-codes L-BFGS although an Adam branch exists; PSO's dependency is undeclared. Port the deployed optimizer selection/configuration and distinguish smoke budgets from scientific runs. Section 2.2. |
| P1 | Local diagnosis is an artifact/log summary | Port deterministic residual/convergence/bound/gradient checks and plotting; ground any Ollama interpretation in those results. Do not describe the present summary as scientific diagnosis. Sections 3.3–3.4. |
| P1 | Local interaction and ingestion are narrower | Missing information causes an error rather than a conversation; text collection cannot extract PDF equations and skips files over 50 KB. Add resumable clarification/document extraction, or describe the actual supported text-input contract. pp.1–3, 7, 10, 13. |
| P1 | Run provenance and historical diagnosis need expansion | Reuse deployed run selection and snapshot behavior. Save data/config/code and seed provenance; analyze historical runs against their snapshots. Initial local timestamp-only directories were insufficient. pp.14–15. |
| P2 | Local live monitoring absent | Port deployed progress/dashboard support if keeping the live URL and monitored-intervention claims. Preserve human approval for changes to running scientific work. p.14. |

Keep the local architectural constraints: YAML, Ollama orchestration, deterministic
acceptance checks, and the existing five-command CLI. Port numerical/runtime code
where possible; adapt Claude procedures into local prompts and deterministic
workflow behavior rather than copying the provider dependency.

The local checker deliberately treats LLM critical claims as unverified semantic
warnings. Retain that distinction and add deterministic checks for demonstrable
faults. A passing check does not certify scientific correctness.

For multi-record objectives, define both weighting levels: equal weight per
experiment, and weighting of channels/finite residuals within an experiment.
A mean of per-channel means is not a pooled residual mean when observation counts
differ.

**Paper edits required for a local-agent release**

| Location | Edit |
|---|---|
| Abstract and Section 1.2, pp.1–3 | Describe Ollama-backed local orchestration and actual human inspection points. Separate supported behavior from intended behavior. |
| Section 2.1, pp.3–4 | Describe local prompt templates, split calls, JSON parsing, deterministic checks and bounded repairs. Replace the note advocating API agents over local deployment. Private operation depends on the configured Ollama endpoint. Avoid claims that prompts guarantee output compliance. |
| Section 2.2.1, pp.4–5 | Current local default is DE. Its implementation uses randomized stratified LHS, whereas the PSO path uses optimized space-filling initialization. Clarify available installation/configuration choices. |
| Section 2.2.2, pp.5–6 | Document the optimizer choices actually exposed locally. Distinguish iteration-budget exhaustion from demonstrated convergence and describe best-point/failure handling accurately. |
| Section 2.3, p.6 | Retain once the deployed analysis is ported and validated locally. Distinguish normalized optimizer coordinates from physical/log10 curvature coordinates. Resolve the relative-floor/identifiable-spread criterion: retained modes above a 1e-6 relative floor cannot themselves span more than six decades. Avoid unconditional identifiability claims at nonstationary or boundary points. |
| Section 3 commands, pp.7–15 | Use terminal commands `pfit new/check/jax/run/diagnose` with `sessions/<name>`, instead of slash skills. Clarify which steps actually call Ollama. |
| Setup, p.7 | Local `requirements.txt` points to `.[test]`; dependencies are in `pyproject.toml` and only partly pinned. Record the evaluated environment, model/quantization, hardware, context length and generation settings. |
| Data/setup, pp.7–13 | Local `new` requires headers, first column `time`, and measured names matching declared states/observables. The tutorial's `time_min` is currently rejected. Keep units in descriptions. Use a single-record tutorial until the deployed multi-record/data contract is ported. |
| Check/JAX, pp.13–14 | Describe deterministic versus semantic findings, current correction coverage, fragment/contract checks and numerical smoke tests. Import success or a finite midpoint loss does not prove scientific translation fidelity. |
| Run/diagnose, pp.14–15 | Keep claims only for features actually ported and tested locally. The deployed reference exists, so reuse it rather than presenting these as speculative new functionality. |
| Examples, p.15 | Report generation success, optimizer execution and scientific fit quality separately for the exact local model/version. Historical evaluations cover different stages and disagree on some statuses; rerun a consistent suite. Verify experimental-data provenance per example. |
| Table 1/Figure 1, p.17 | Replace stale XML/skeleton descriptions with text/CSV inputs and generated YAML/model/JAX artifacts. Distinguish orchestration `pfit.yaml` from session numerical settings. |
| Throughout | Resolve author/contact placeholders, citation TODOs, broken subsection formatting, and missing table/figure/section references. External bibliography verification is outside this review. |

**Verification**

The initial source review and temporary probes established local defects:
multiple configured experiments select only the first; decreasing time passes
preflight; default-loss handling can replace custom writeout; and missing
measurements contaminate standard-loss normalization. These local findings remain
valid despite the corrected reference branch. The original branch comparison was
corrected after inspecting the deployed tree, drivers, numerical helpers,
sloppiness implementation, run store and tests. Feature-port validation is
recorded separately in the restart/sloppiness document.

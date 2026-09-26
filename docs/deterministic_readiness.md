Deterministic readiness

Implemented 2026-09-26 against `pfit-claude origin/deployed_branch` at `1b415d8`,
using `tools/check_ready.py`, `lib/utils/source_stamp.py`, and the deployed
validation rules as references.

`pfit check SESSION --deterministic-only` runs input/model checks without creating
an Ollama client. `--ready` also requires generated code with the expected
interface and current model/YAML sources. `pfit run` invokes the same gate before
importing the numerical engine or creating a run directory. Gradient-only runs
also retain completed-seed selection, parameter-name/order and bounds validation.
`--ready` itself checks the shared prerequisites, not a selected restart seed.

Checks cover CSV shape and declared width, finite strictly increasing times,
initial-time coverage, finite ordered bounds, positive log bounds, actual boolean
logscale flags, finite fixed values/initial conditions, identifiers, integer
budgets, population minimums, positive finite scalar/per-state tolerances,
timestep and error-loss settings, Python syntax, and literal dataset/solution
column indices. Checking, fitting, smoke testing and snapshot reanalysis share
numeric CSV loading. Missing measurements are retained; malformed numeric cells
and repeated header rows are rejected.

After generated-code contract validation and the numerical smoke test pass,
`pfit jax` stamps model and YAML content hashes into the script. Sources changing
during generation invalidate acceptance. Failed scripts have a pending marker
and cannot pass as legacy scripts. Stamped mismatches block fitting even when
file timestamps match. Unstamped historical scripts retain the reference's
mtime fallback, explicitly warning that copies/clones can make it unreliable.

Local adaptations:

- Hash whole source files, preserving text after `#` inside strings. Comment-only
  changes and optimizer-budget edits conservatively require retranslation too.
- Rerun current deterministic checks instead of trusting an old saved report or
  an LLM critical-error assertion. Reports include an explicit critical count.
- Allow tiny numerical budgets for smoke tests; only invalid settings block.
- Multi-experiment follow-up: validate every record and its initial-condition
  overrides, enforce a common column schema, and smoke-test every record before
  stamping accepted code. See [implementation notes](multi_experiment_support.md).

Remaining limits: this is not full numerical-equivalence/gradient validation,
complete static shape inference, forcing support, or a complete missing-data and
uncertainty audit. Source stamps record provenance, not proof of translated
mathematical equivalence. The generation smoke test remains a numerical check
at one parameter point per experiment. Multi-experiment support is now implemented
within the common-column/shared-parameter scope recorded under OPEN-01.

Sneyd IPR is the correct next reference example: the deployed YAML declares nine
CSV files and overrides the zero-derivative IP3/Ca states per experiment. Boehm's
prepared YAML declares one experiment with three observable columns.

Validation: 243 default tests passed, 4 deselected. A separate targeted run passed
47 tests, including translation workflows and the full-fit → gradient-only
restart → snapshot reanalysis numerical regression. A fresh-process CLI check
confirmed invalid input is rejected without importing JAX. `git diff --check`
passed. No live Ollama integration was run.

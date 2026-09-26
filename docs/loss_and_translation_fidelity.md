Loss and translation fidelity

User functions are authoritative. A plain MSE function is no longer inferred to
be a framework default and silently replaced by normalized RMSE. Unmarked
constant losses are also preserved. Only explicitly marked, unchanged skeleton
functions may receive a framework default; the marker alone is insufficient if
the function was edited. Older unmarked stubs must be completed by the user.

Loss and writeout are selected independently. Straight-line losses retain their
operations through NumPy-to-JAX conversion; more complex losses use the LLM
translation path. Custom writeout is preserved as host NumPy code when supported,
including ordering, transforms and additional columns. Legacy write_problem_result
source functions are recognized as well as writeout_description. JAX RHS/helpers
still use Ollama translation. The user-defined per-experiment loss is averaged
equally across experiments; no new normalization or weighting is introduced.

Before stamping generated code as accepted, numerical probes compare source
Python and generated JAX for every experiment at normalized parameter offsets
0 and 0.1. RHS values are compared at initial/middle/final simulated states; loss
and full writeout shapes/values are compared on the same trajectories. The
threshold is rtol=1e-6, atol=1e-8. Matching NaN display positions are allowed in
writeout, but invalid simulated trajectories or nonfinite losses are not.
A failed secondary parameter solve is recorded as a limitation; midpoint checks
must pass. Unchanged skeleton components have no user-supplied semantics and are
explicitly excluded. Missing components/interfaces are recorded in the report.

A mismatch blocks source stamping and feeds the existing bounded JAX repair loop
with the experiment, component, probe offset, shapes and maximum difference.
The model must repair its translation, not edit source intent. Numerical evidence
is saved to generated/translation_fidelity.json. A user-edited source requires
fresh translation/readiness, as before.

This is sampled equivalence against the supplied Python model, not proof across
all parameters/states and not verification against a paper. Extraction may still
produce a scientifically incorrect Python model; the user's review remains
necessary. These probes can now catch the kind of subsequent JAX coefficient
transcription error encountered in Sneyd without needing an external oracle.

Historical fits and source models, including the existing Boehm objective
mismatch, are not silently rewritten by this change. Correct those source
objectives explicitly and refit before comparing numerical results.

Validation: the full suite passed 288 tests (7 deselected), followed by the
expanded targeted preservation suite. Tests deliberately substitute RMSE for
MSE, alter an RHS coefficient, and reorder writeout columns: all are rejected
and repaired, or remain unstamped if repair is exhausted. Raw MSE and a custom
writeout are preserved; an actual default loss does not replace custom output.

Live qwen2.5-coder:32b passed check/JAX (including numerical acceptance),
independent reference-loss comparison, bounded fitting and Ollama-backed diagnosis
on cascaded tanks. Artifacts: evaluation_runs/live_fidelity_cascaded_20260926/.

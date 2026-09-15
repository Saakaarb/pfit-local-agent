---
topic: How to diagnose autodiff failures in pfit gradient refinement
consumed_by: [pfit-diagnose, autodiff_diagnose.py, agents fixing failed fits]
generated: false
owns: >
  The theory-to-action bridge for neural-ODE/ODE autodiff failures: adjoint
  choice, solver family, tolerances, non-smooth models, event-like losses, and
  local diagnostic probes.
---

# Autodiff diagnosis rules

Use this after a fit reaches the gradient/NODE stage and then exits with a JAX,
Equinox, Diffrax, Lineax, NaN, or non-finite gradient failure. This is not a
replacement for `tuning_rules.md`; it explains why a forward solve can be good
enough for population search while differentiation through that solve fails.

## Why ODE autodiff fails

Diffrax offers several adjoint methods. Its default,
`RecursiveCheckpointAdjoint`, differentiates the numerical solution itself
("discretise then optimise") and requires a finite `max_steps` budget so the
checkpointed computation can be represented by JAX. `BacksolveAdjoint` instead
solves continuous adjoint equations backwards in time; the Diffrax docs do not
recommend it as the default because its gradients are approximate and
checkpointing already gives low memory use.

Neural-ODE papers and tool docs point to the same practical causes:

1. **Stiffness.** The backward/adjoint computation can be at least as sensitive
   as the forward solve. A parameter point that integrates forward may still
   produce exploding, inaccurate, or non-finite sensitivities.
2. **Non-smooth dynamics or losses.** Hard triggers, `where`, `clip`, `abs`,
   `sign`, argmin/argmax event extraction, and thresholded penalties create
   undefined or discontinuous derivatives. These are common in physical models
   with contact, saturation, ignition, threshold crossing, and piecewise
   regimes.
3. **Tolerance noise.** Adaptive solvers select steps by an error controller.
   Forward residuals may be visually accurate while gradients are dominated by
   discretisation or branch changes in the solver path.
4. **Implicit-solver singularity.** Stiff solvers solve nonlinear/linear systems
   inside each step. If those internal systems become ill-conditioned, autodiff
   may surface a Lineax/Equinox error such as "linear solver returned
   non-finite output."
5. **Parameter-space walls.** Hard clipping to bounds, invalid parameter
   regions, or `error_loss` plateaus make the local objective non-smooth even if
   the RHS is smooth.

## Required local probe

Run:

```bash
./venv/bin/python3 tools/autodiff_diagnose.py <session-name> --run <run-id>
```

Read `autodiff_diagnosis.txt` before editing. It tests the completed run, not a
new fit:

- base loss at the final design point;
- `jax.value_and_grad` at the final design point;
- centered finite differences in scaled parameter space;
- nearby axis/random perturbations for failed, non-finite, or `error_loss`
  evaluations;
- static scan for non-smooth expressions in the generated model/loss;
- run logs for known gradient-stage failure signatures.

## Interpretation

### A1 -- `value_and_grad` raises

If the error mentions Lineax, Equinox, a non-finite linear solve, or
`CpuCallback`, then the differentiated solve is failing before the optimizer has
a chance to move.

Fix order:

1. If the static scan found hard switches or event-like scalar losses, smooth
   them or remove them from the differentiated objective.
2. If the RHS is smooth and stiff, keep a stiff solver but try slightly looser
   gradient tolerances first, then tighten until finite-difference and autodiff
   gradients agree.
3. If the RHS is non-smooth but not truly stiff, try an explicit adaptive solver
   from `lib/LLM/api/diffrax.md`.
4. If nearby perturbations hit `error_loss`, narrow the search box around
   physically integrable values before trusting gradient refinement.

### A2 -- gradient finite, finite difference disagrees

This indicates an unreliable derivative, not necessarily a wrong optimum.

Likely causes:

- solver tolerances too loose for sensitivities;
- a discontinuity in the loss or model;
- an adaptive-step branch change dominating the finite-difference window;
- a parameter axis badly scaled relative to the others.

Try two finite-difference scales (`--eps 1e-4` and `--eps 1e-5`). If both
disagree with autodiff, fix smoothness/tolerances before changing the optimizer.

### A3 -- nearby probes fail or hit `error_loss`

The final point is close to a numerical wall. Gradient optimizers assume a
locally smooth neighbourhood, so a good scalar loss at the center is not enough.

Fix:

- narrow bounds that admit unintegrable values;
- increase `max_steps` only if failures are `max_steps_reached` and the solver
  family is otherwise right;
- avoid making `error_loss` so close to feasible losses that the optimizer can
  confuse a wall for a basin.

### A4 -- static scan finds hard events

Never differentiate a hard event time extracted by `argmax`, boolean threshold,
or a discontinuous branch and expect a stable gradient. Prefer a smooth
surrogate:

- logistic/tanh switch over a physically meaningful width;
- softplus instead of ReLU-like kinks;
- differentiable residual on the trajectory near the event, rather than an
  argmin/first-crossing scalar;
- population-only treatment for truly discrete observables.

## Sources used for this rule

- Diffrax adjoint documentation: default `RecursiveCheckpointAdjoint`, forward
  mode option, and cautions against `BacksolveAdjoint`.
- Diffrax issue guidance: finite `max_steps` is required for default
  checkpointed backpropagation because JAX cannot allocate dynamic memory during
  the differentiated solve.
- Chen et al. (2018), *Neural Ordinary Differential Equations*: popularised the
  continuous-adjoint training formulation.
- Kim et al. (2021), *Adaptive Checkpoint Adjoint Method for Gradient Estimation
  in Neural ODE*: discusses truncation/reverse-trajectory error in adjoint
  gradients and deep solver computation graphs.
- Raue/Fröhlich-style sensitivity literature in systems biology: gradient
  accuracy depends materially on flow and adjoint integration tolerances,
  especially in stiff models.
- torchdiffeq FAQ: adaptive tolerances control accepted steps, stiffness can
  drive step-size underflow, and non-smooth nonlinearities should be avoided for
  neural ODE training.

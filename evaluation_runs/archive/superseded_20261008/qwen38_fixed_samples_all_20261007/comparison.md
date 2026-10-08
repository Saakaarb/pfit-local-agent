# Live model comparison

Single attempt per model/case, including the existing repair budget. Times include failed attempts and must be interpreted alongside status. Downloads and warm-up are recorded separately. These are workflow smoke tests, not converged fits.

| Case | qwen38_27b (status; seconds) |
|---|---|
| ARC_fitting | pass; 620.018 |
| boehm_stat5 | pass; 463.132 |
| hodgkin_huxley | blocked; 0 |
| lotka_volterra | pass; 274.647 |
| mapk_cascade | fail at jax; 319.311 |
| nfkb_signaling | fail at new; 168.618 |
| oregonator | fail at jax; 289.877 |
| piezo_bouc_wen | pass; 326.306 |
| robertson_session | pass; 285.93 |
| session1 | blocked; 0 |
| sliding_basepoint_headered | fail at jax; 329.633 |
| test_session | pass; 297.741 |
| theophylline | pass; 223.501 |
| vanderpol_session | pass; 213.881 |
| cascaded_tanks | pass; 400.779 |
| decay_multiexp | pass; 272.719 |
| sneyd_ipr | pass; 463.458 |
| beer_indigoidine | pass; 250.156 |
| raia_il13 | fail at jax; 369.812 |
| schwen_insulin | pass; 530.781 |
| armistead_sphingolipid | pass; 387.522 |
| borghans_calcium | fail at run; 234.795 |
| fujita_egf | fail at jax; 483.447 |

## Counts

| Model | Pass | Fail/degraded | Infrastructure/blocked | Pending/running |
|---|---:|---:|---:|---:|
| qwen38_27b | 14 | 7 | 2 | 0 |

Two incomplete folders are excluded from the eligible denominator. test_session duplicates Robertson; sliding_basepoint_headered duplicates sliding_basepoint. Keep these rows visible but do not treat them as independent scientific problems.

A pass requires all five stages, the declared experiment files/count, finite result arrays and final loss, completed refinement, and accepted Ollama diagnosis. The framework applies semantic checks and source-to-JAX fidelity checks. This is not an independent proof that every generated equation matches the original scientific specification.

Models differ in architecture, generation, and quantization. Dense total parameters are used as nominal active counts; active counts for MoE models are approximate. Thinking capability and requested mode are recorded in model metadata. For thinking models, the generation budget also includes reasoning tokens. One trial cannot estimate success probabilities or timing variance.

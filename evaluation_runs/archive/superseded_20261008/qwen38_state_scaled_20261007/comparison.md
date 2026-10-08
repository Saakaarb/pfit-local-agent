# Live model comparison

Single attempt per model/case, including the existing repair budget. Times include failed attempts and must be interpreted alongside status. Downloads and warm-up are recorded separately. These are workflow smoke tests, not converged fits.

| Case | qwen38_27b (status; seconds) |
|---|---|
| ARC_fitting | pass; 438.218 |
| boehm_stat5 | pending |
| hodgkin_huxley | pending |
| lotka_volterra | pending |
| mapk_cascade | pending |
| nfkb_signaling | pending |
| oregonator | pending |
| piezo_bouc_wen | pending |
| robertson_session | pending |
| session1 | pending |
| sliding_basepoint_headered | pending |
| test_session | pending |
| theophylline | pass; 241.816 |
| vanderpol_session | pending |
| cascaded_tanks | pending |
| decay_multiexp | pending |
| sneyd_ipr | pending |
| beer_indigoidine | pending |
| raia_il13 | pending |
| schwen_insulin | pending |
| armistead_sphingolipid | pass; 384.132 |
| borghans_calcium | pending |
| fujita_egf | pending |

## Counts

| Model | Pass | Fail/degraded | Infrastructure/blocked | Pending/running |
|---|---:|---:|---:|---:|
| qwen38_27b | 3 | 0 | 0 | 20 |

Two incomplete folders are excluded from the eligible denominator. test_session duplicates Robertson; sliding_basepoint_headered duplicates sliding_basepoint. Keep these rows visible but do not treat them as independent scientific problems.

A pass requires all five stages, the declared experiment files/count, finite result arrays and final loss, completed refinement, and accepted Ollama diagnosis. The framework applies semantic checks and source-to-JAX fidelity checks. This is not an independent proof that every generated equation matches the original scientific specification.

Models differ in architecture, generation, and quantization. Dense total parameters are used as nominal active counts; active counts for MoE models are approximate. Thinking capability and requested mode are recorded in model metadata. For thinking models, the generation budget also includes reasoning tokens. One trial cannot estimate success probabilities or timing variance.

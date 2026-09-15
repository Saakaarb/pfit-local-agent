import math

from tools import autodiff_diagnose


def test_scan_model_filters_missing_data_masks_but_keeps_event_logic(tmp_path):
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "user_model.py").write_text(
        """
import numpy as np

def _loss(solution_time, solution, dataset):
    F_exp = dataset[:, 0]
    valid_F = np.isfinite(F_exp)
    F_safe = np.where(valid_F, F_exp, 0.0)
    above = solution[:, 0] >= 10.0
    idx = np.argmax(above)
    return np.where(np.any(above), solution_time[idx], solution_time[-1])
"""
    )
    (generated / "generated_script.py").write_text("")

    hits = autodiff_diagnose._scan_model(tmp_path)
    texts = [hit["text"] for hit in hits]

    assert not any("F_safe" in text for text in texts)
    assert any("argmax" in text for text in texts)
    assert any(">=" in text for text in texts)
    assert any("np.where(np.any(above)" in text for text in texts)


def test_recommend_names_nonfinite_gradient_axes():
    results = {
        "autodiff_gradient": {
            "ok": False,
            "error": "non-finite value or gradient",
            "nonfinite_axes": ["E1", "E2"],
        },
        "static_nonsmooth_hits": [],
        "perturbation_probe": {
            "counts": {"failed": 0, "penalized": 0, "nonfinite": 0}
        },
        "logs": {},
        "gradient_vs_finite_difference": None,
    }

    recommendations = autodiff_diagnose._recommend(results)

    assert any("E1, E2" in rec for rec in recommendations)


def test_json_default_handles_numpy_like_scalars():
    assert autodiff_diagnose._json_default(math.inf) == "inf"

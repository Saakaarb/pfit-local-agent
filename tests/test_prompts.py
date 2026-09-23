import pytest

from local_agent.agent.prompts import PromptRenderer


pytestmark = pytest.mark.unit


def test_prompt_renderer_substitutes_context():
    renderer = PromptRenderer()

    text = renderer.render(
        "jax_fragments.user.md",
        {
            "input_yaml": "model: {}",
            "session_summary": "0: mu",
            "user_model": "def model(): pass",
        },
    )

    assert "model: {}" in text
    assert "0: mu" in text
    assert "def model(): pass" in text


def test_jax_fragments_system_prompt_contains_fragment_contract():
    renderer = PromptRenderer()

    text = renderer.render("jax_fragments.system.md", {})

    assert '"rhs"' in text
    assert '"loss_body"' in text
    assert "Do not write generated_script.py" in text
    assert "The framework removes the time column" in text
    assert "Preserve dataset indices from user_model.py" in text
    assert "Preserve dataset-column transforms exactly" in text
    assert "Do not change MSE into RMSE" in text
    assert "1000.0 * dataset[:, 0]" in text


def test_new_session_parameters_prompt_excludes_initial_conditions():
    renderer = PromptRenderer()

    text = renderer.render("new_session_parameters.system.md", {})

    assert "Initial conditions" in text
    assert "not fixed parameters" in text


def test_new_session_loss_prompt_uses_log_for_large_positive_range():
    renderer = PromptRenderer()

    text = renderer.render("new_session_loss.system.md", {})

    assert "3 log10 orders" in text
    assert "log10_normalized_mse" in text


def test_repair_jax_fragments_system_prompt_rejects_framework_plumbing():
    renderer = PromptRenderer()

    text = renderer.render("repair_jax_fragments.system.md", {})

    assert "Repair only the field implicated by the validation error" in text
    assert "Do not include framework plumbing" in text
    assert "Preserve dataset indices from" in text
    assert "assigned earlier in that same body" in text
    assert "Preserve dataset-column transforms exactly" in text
    assert "Copy every other field" in text
    assert "Do not regenerate valid fields" in text


def test_split_body_prompts_require_local_name_checks():
    renderer = PromptRenderer()

    loss_text = renderer.render("jax_loss.system.md", {})
    writeout_text = renderer.render("jax_writeout.system.md", {})

    assert "assigned earlier in loss_body" in loss_text
    assert "Preserve dataset-column transforms exactly" in loss_text
    assert "Do not change MSE into RMSE" in loss_text
    assert "1000.0 * dataset[:, 0]" in loss_text
    assert "assigned earlier in writeout_body" in writeout_text
    assert "Preserve the user's writeout columns" in writeout_text
    assert "Preserve dataset-column transforms exactly" in writeout_text
    assert "do not replace derived" in writeout_text
    assert "not differentiated and is not jitted" in writeout_text
    assert "NumPy as np" in writeout_text


def test_jax_user_prompts_end_with_json_only_instruction():
    renderer = PromptRenderer()

    context = {
        "input_yaml": "model: {}",
        "session_summary": "summary",
        "helper_function_inventory": "- none",
        "rhs_intermediate_inventory": "- flux = k * x",
        "user_model": "def user_defined_system(): pass",
    }

    text = renderer.render("jax_fragments.user.md", context)
    repair_text = renderer.render(
        "repair_jax_fragments.user.md",
        {
            **context,
            "attempt": "1",
            "validation_error": "bad",
            "previous_response": "{}",
        },
    )

    assert "first character of your response must be `{`" in text
    assert "Do not explain the model" in text
    assert "first character of your response must be `{`" in repair_text
    assert '"rhs": []' in text
    assert "previous_response" in repair_text
    assert "copy all other fields exactly" in repair_text
    assert "Rendered generated_script.py" not in repair_text
    assert "RHS intermediates from user_model.py" in text
    assert "Do not use these intermediate names as bare values" in repair_text

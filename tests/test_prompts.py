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


def test_repair_jax_fragments_system_prompt_rejects_framework_plumbing():
    renderer = PromptRenderer()

    text = renderer.render("repair_jax_fragments.system.md", {})

    assert "Repair only the fragment fields" in text
    assert "Do not include framework plumbing" in text
    assert "Preserve dataset indices from" in text


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
            "generated_script": "",
        },
    )

    assert "first character of your response must be `{`" in text
    assert "Do not explain the model" in text
    assert "first character of your response must be `{`" in repair_text
    assert '"rhs": []' in text
    assert '"loss_body": ""' in repair_text
    assert "RHS intermediates from user_model.py" in text
    assert "Do not use these intermediate names as bare values" in repair_text

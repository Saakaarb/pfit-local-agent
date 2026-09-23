from pathlib import Path

import pytest

from local_agent.agent.session_spec import load_session_spec


pytestmark = pytest.mark.unit


def test_load_session_spec_preserves_ordering():
    spec = load_session_spec(Path("tests/robertson_session/inputs/user_input.yaml"))

    assert [param.name for param in spec.trainable_parameters] == ["k1", "k2", "k3"]
    assert [var.name for var in spec.integrated_variables] == ["y1", "y2", "y3"]
    assert spec.fixed_parameters[0].name == "unused_constant"
    assert spec.integrator == "Kvaerno5"


def test_session_spec_prompt_text_contains_vector_ordering():
    spec = load_session_spec(Path("tests/vanderpol_session/inputs/user_input.yaml"))

    prompt_text = spec.to_prompt_text()

    assert "Trainable parameter order:" in prompt_text
    assert "0: mu" in prompt_text
    assert "Integrated variable order:" in prompt_text
    assert "0: x1" in prompt_text
    assert "1: x2" in prompt_text
    assert "Runtime dataset columns after removing time:" in prompt_text
    assert "dataset[:, 0]: x1" in prompt_text
    assert "dataset[:, 1]: x2" in prompt_text
    assert "Integrator: " in prompt_text

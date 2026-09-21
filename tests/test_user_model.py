from pathlib import Path

import pytest

from local_agent.agent.session_spec import load_session_spec
from local_agent.agent.user_model import render_user_model_skeleton


pytestmark = pytest.mark.unit


def test_render_user_model_skeleton_uses_yaml_ordering():
    spec = load_session_spec(Path("tests/vanderpol_session/inputs/user_input.yaml"))

    source = render_user_model_skeleton(spec)

    assert "mu = trainable_parameters['mu']" in source
    assert "x1 = y[0]" in source
    assert "x2 = y[1]" in source
    assert "dx1dt = 0.0" in source
    assert "dx2dt = 0.0" in source
    assert "writeout_array = np.zeros([solution_time.shape[0], 5])" in source

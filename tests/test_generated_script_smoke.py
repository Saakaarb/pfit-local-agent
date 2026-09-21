from pathlib import Path

import pytest

from local_agent.agent.validators import ValidationError, smoke_test_generated_script


pytestmark = pytest.mark.contract


VALID_SCRIPT = """
import numpy as np

def user_defined_system(t, y, other_args):
    return y

def _integrate_system(constants, trainable_variables):
    return None

def _compute_loss_problem(constants, trainable_variables):
    return 0.0

def _write_problem_result(constants, trainable_variables):
    rows = constants["dataset"].shape[0]
    return np.zeros((rows, 2))
"""


def make_session(tmp_path):
    session = tmp_path / "session"
    inputs = session / "inputs"
    inputs.mkdir(parents=True)
    yaml_source = (Path("tests/vanderpol_session/inputs/user_input.yaml")).read_text()
    data_source = (Path("tests/vanderpol_session/inputs/vanderpol_data.csv")).read_text()
    (inputs / "user_input.yaml").write_text(yaml_source)
    (inputs / "vanderpol_data.csv").write_text(data_source)
    return session


def test_smoke_test_generated_script_passes(tmp_path):
    session = make_session(tmp_path)
    script = tmp_path / "generated_script.py"
    script.write_text(VALID_SCRIPT)

    smoke_test_generated_script(script, session)


def test_smoke_test_rejects_nonfinite_loss(tmp_path):
    session = make_session(tmp_path)
    script = tmp_path / "generated_script.py"
    script.write_text(VALID_SCRIPT.replace("return 0.0", "return float('nan')"))

    with pytest.raises(ValidationError, match="non-finite loss"):
        smoke_test_generated_script(script, session)


def test_smoke_test_rejects_wrong_writeout_shape(tmp_path):
    session = make_session(tmp_path)
    script = tmp_path / "generated_script.py"
    script.write_text(VALID_SCRIPT.replace("return np.zeros((rows, 2))", "return np.zeros(2)"))

    with pytest.raises(ValidationError, match="2D array"):
        smoke_test_generated_script(script, session)

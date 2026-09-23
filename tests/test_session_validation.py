from pathlib import Path

import pytest

from local_agent.agent.validators import ValidationError, parse_input_yaml, validate_session


pytestmark = pytest.mark.unit


FIXTURES = Path("tests")


def test_validate_existing_robertson_session():
    result = validate_session(FIXTURES / "robertson_session")

    assert result.dataset_shape[1] == 4
    assert result.n_trainable_parameters == 3
    assert result.n_integrated_variables == 3


def test_parse_input_yaml_rejects_duplicate_names(tmp_path):
    yaml_path = _write_yaml(tmp_path, _minimal_yaml(variable_name="k"))

    with pytest.raises(ValidationError, match="not unique"):
        parse_input_yaml(yaml_path)


def test_validate_session_rejects_missing_dataset(tmp_path):
    yaml_path = _write_yaml(tmp_path, _minimal_yaml(data_file="missing.csv"))
    assert yaml_path.exists()

    with pytest.raises(ValidationError, match="Dataset file not found"):
        validate_session(tmp_path / "session")


def test_validate_session_allows_observed_transforms_or_state_subsets(tmp_path):
    yaml_path = _write_yaml(tmp_path, _minimal_yaml())
    yaml_path.parent.joinpath("data.csv").write_text("0.0,1.0\n1.0,2.0\n")

    result = validate_session(tmp_path / "session")

    assert result.dataset_shape == (2, 2)
    assert result.n_integrated_variables == 1


def test_parse_input_yaml_defaults_to_de(tmp_path):
    reader = parse_input_yaml(_write_yaml(tmp_path, _minimal_yaml()))

    assert reader.algorithm == "DE"
    assert reader.random_seed is None


def test_parse_input_yaml_accepts_de_algorithm_and_seed(tmp_path):
    reader = parse_input_yaml(
        _write_yaml(tmp_path, _minimal_yaml(algorithm="DE", random_seed=123))
    )

    assert reader.algorithm == "DE"
    assert reader.random_seed == 123


def test_parse_input_yaml_accepts_population_tolerances(tmp_path):
    reader = parse_input_yaml(
        _write_yaml(
            tmp_path,
            _minimal_yaml(population_rtol="[1e-4]", population_atol="[1e-6]"),
        )
    )

    assert reader.population_stepsize_rtol == [1e-4]
    assert reader.population_stepsize_atol == [1e-6]
    assert reader.pso_stepsize_rtol == [1e-4]
    assert reader.pso_stepsize_atol == [1e-6]


def test_parse_input_yaml_rejects_unknown_population_algorithm(tmp_path):
    yaml_path = _write_yaml(tmp_path, _minimal_yaml(algorithm="CMAES"))

    with pytest.raises(ValidationError, match="Unsupported population_opt algorithm"):
        parse_input_yaml(yaml_path)


def test_parse_input_yaml_accepts_configured_integrator(tmp_path):
    reader = parse_input_yaml(
        _write_yaml(tmp_path, _minimal_yaml(integrator="Kvaerno5"))
    )

    assert reader.integrator == "Kvaerno5"


def test_parse_input_yaml_rejects_unknown_integrator(tmp_path):
    yaml_path = _write_yaml(tmp_path, _minimal_yaml(integrator="BogusSolver"))

    with pytest.raises(ValidationError, match="Unsupported gradient_opt integrator"):
        parse_input_yaml(yaml_path)


def test_validate_session_allows_nan_measurements(tmp_path):
    yaml_path = _write_yaml(tmp_path, _minimal_yaml())
    yaml_path.parent.joinpath("data.csv").write_text("0.0,1.0,0.0\n1.0,nan,0.1\n")

    result = validate_session(tmp_path / "session")

    assert result.dataset_shape == (2, 3)


def _write_yaml(tmp_path: Path, text: str) -> Path:
    inputs = tmp_path / "session" / "inputs"
    inputs.mkdir(parents=True)
    path = inputs / "user_input.yaml"
    path.write_text(text)
    return path


def _minimal_yaml(
    *,
    data_file: str = "data.csv",
    parameter_name: str = "k",
    variable_name: str = "y",
    algorithm: str = "DE",
    random_seed: int | None = None,
    population_rtol: str = "null",
    population_atol: str = "null",
    integrator: str = "Tsit5",
) -> str:
    random_seed_line = "" if random_seed is None else f"  random_seed: {random_seed}\n"
    return f"""experiments:
  - data_file: {data_file}
    columns:
      - {{name: time}}
      - {{name: obs, observes: {variable_name}}}

model:
  trainable_parameters:
    - {{name: {parameter_name}, min_val: 0.1, max_val: 10.0, logscale: true}}
  fixed_parameters: []
  integrated_variables:
    - {{name: {variable_name}, init_val: 1.0}}

population_opt:
  population_size: 4
  num_iters: 1
  processors: 1
  algorithm: {algorithm}
{random_seed_line}  stepsize_rtol: {population_rtol}
  stepsize_atol: {population_atol}

gradient_opt:
  num_iters: 1
  stepsize_rtol: [1e-7]
  stepsize_atol: [1e-9]
  initial_timestep: 1e-6
  max_steps: 100
  integrator: {integrator}
  init_value_lr: 1e-4
  end_value_lr: 1e-5
  transition_steps_lr: 10
  decay_rate_lr: 0.9

output:
  write_results: true
"""

"""Create the paper pulse/washout session for real pfit run/diagnose outputs."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import yaml

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))


ROOT = Path("docs/paper_running_example")
SESSION = ROOT / "session"
TRUE = dict(k_elim=0.42, k_on=0.85, k_off=0.18, baseline=92.0, gain=54.0)


def forcing(kind: str, t: np.ndarray) -> np.ndarray:
    if kind == "pulse":
        return np.where((t >= 2.0) & (t <= 9.0), 0.38, 0.0)
    if kind == "washout":
        return np.where(t <= 5.0, 0.16, 0.0)
    raise ValueError(kind)


def simulate(params: dict[str, float], t_eval: np.ndarray, dose: np.ndarray, y0: tuple[float, float]) -> np.ndarray:
    def rhs(t, y):
        dose_rate = np.interp(t, t_eval, dose)
        D, R = y
        return np.asarray([
            dose_rate - params["k_elim"] * D,
            params["k_on"] * D * (1.0 - R) - params["k_off"] * R,
        ])

    states = np.zeros((len(t_eval), 2), dtype=float)
    states[0] = y0
    for i in range(1, len(t_eval)):
        dt = float(t_eval[i] - t_eval[i - 1])
        t0 = float(t_eval[i - 1])
        y = states[i - 1]
        k1 = rhs(t0, y)
        k2 = rhs(t0 + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = rhs(t0 + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = rhs(t0 + dt, y + dt * k3)
        states[i] = y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    signal_model = params["baseline"] + params["gain"] * states[:, 1]
    return np.column_stack([states, signal_model])


def make_inputs() -> None:
    inputs = SESSION / "inputs"
    generated = SESSION / "generated"
    inputs.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)

    user_info = """Problem:
Fit pulse.csv and washout.csv using one shared parameter set.

Experiments and initial conditions:
pulse.csv: D(0) = 0, R(0) = 0
washout.csv: D(0) = 0, R(0) = 0.3

CSV columns:
time: time in minutes
dose_rate: known forcing input in concentration per minute; use linear interpolation
signal: measured value of signal_model in arbitrary units

Trainable parameters:
k_elim: bounds [0.01, 10], logarithmic
k_on: bounds [0.001, 10], logarithmic
k_off: bounds [0.001, 10], logarithmic
baseline: bounds [50, 150], linear
gain: bounds [1, 200], linear

Fixed parameters:
None.

ODE system:
dD/dt = dose_rate - k_elim*D
dR/dt = k_on*D*(1 - R) - k_off*R

Observable:
signal_model = baseline + gain*R
CSV column signal measures signal_model.

Loss:
For each experiment e, use only finite signal measurements.
scale_e = max(abs(finite signal values in experiment e))
L_e = sqrt(mean(((signal_model_e - signal_e)/scale_e)^2))
The total loss is the equal-weight mean: L = (L_pulse + L_washout)/2.

Outputs:
Write time, D, R, measured signal, and signal_model.
"""
    (inputs / "user_info.txt").write_text(user_info)

    rng = np.random.default_rng(7)
    for index, (name, y0) in enumerate((("pulse", (0.0, 0.0)), ("washout", (0.0, 0.3)))):
        t = np.linspace(0.0, 24.0, 49)
        dose = forcing(name, t)
        signal_model = simulate(TRUE, t, dose, y0)[:, 2]
        signal = signal_model + rng.normal(0.0, 0.85, size=t.shape) + 0.35 * np.sin(0.55 * t + index)
        np.savetxt(inputs / f"{name}.csv", np.column_stack([t, dose, signal]),
                   delimiter=",", header="time,dose_rate,signal", comments="")

    config = {
        "experiments": [
            {"data_file": "pulse.csv", "initial_conditions": {}, "columns": [
                {"name": "time"}, {"name": "dose_rate", "role": "forcing", "interpolation": "linear"},
                {"name": "signal", "observes": "signal_model"}]},
            {"data_file": "washout.csv", "initial_conditions": {"R": 0.3}, "columns": [
                {"name": "time"}, {"name": "dose_rate", "role": "forcing", "interpolation": "linear"},
                {"name": "signal", "observes": "signal_model"}]},
        ],
        "model": {
            "trainable_parameters": [
                {"name": "k_elim", "min_val": 0.01, "max_val": 10.0, "logscale": True},
                {"name": "k_on", "min_val": 0.001, "max_val": 10.0, "logscale": True},
                {"name": "k_off", "min_val": 0.001, "max_val": 10.0, "logscale": True},
                {"name": "baseline", "min_val": 50.0, "max_val": 150.0, "logscale": False},
                {"name": "gain", "min_val": 1.0, "max_val": 200.0, "logscale": False},
            ],
            "fixed_parameters": [],
            "integrated_variables": [{"name": "D", "init_val": 0.0}, {"name": "R", "init_val": 0.0}],
            "observables": [{"name": "signal_model"}],
        },
        "population_opt": {"population_size": 48, "num_iters": 120, "processors": 1, "algorithm": "DE"},
        "gradient_opt": {
            "gradient_optimizer": "adam", "num_iters": 2000, "stepsize_rtol": [1e-7, 1e-7],
            "stepsize_atol": [1e-9, 1e-9], "initial_timestep": 1e-4, "max_steps": 10000,
            "integrator": "Tsit5", "init_value_lr": 1e-3, "end_value_lr": 1e-5,
            "transition_steps_lr": 1000, "decay_rate_lr": 0.9,
        },
        "output": {"write_results": True},
    }
    (inputs / "user_input.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


def write_user_model() -> None:
    (SESSION / "generated" / "user_model.py").write_text(
        """import numpy as np

def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    k_elim = trainable_parameters['k_elim']
    k_on = trainable_parameters['k_on']
    k_off = trainable_parameters['k_off']
    D = y[0]
    R = y[1]
    dose_rate = np.interp(t, t_eval, dataset[:, 0])
    return np.array([dose_rate - k_elim * D, k_on * D * (1.0 - R) - k_off * R])

def _observables(solution, trainable_parameters, fixed_parameters):
    return {'signal_model': trainable_parameters['baseline'] + trainable_parameters['gain'] * solution[:, 1]}

def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    signal_model = trainable_parameters['baseline'] + trainable_parameters['gain'] * solution[:, 1]
    signal = dataset[:, 1]
    mask = np.isfinite(signal)
    scale = np.max(np.abs(signal[mask]))
    residual = (signal_model[mask] - signal[mask]) / scale
    return float(np.sqrt(np.mean(residual * residual)))

def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    signal_model = trainable_parameters['baseline'] + trainable_parameters['gain'] * solution[:, 1]
    writeout_array = np.zeros([solution_time.shape[0], 5])
    writeout_array[:, 0] = solution_time
    writeout_array[:, 1] = solution[:, 0]
    writeout_array[:, 2] = solution[:, 1]
    writeout_array[:, 3] = dataset[:, 1]
    writeout_array[:, 4] = signal_model
    return writeout_array
"""
    )


def write_generated_script() -> None:
    from local_agent.agent.jax_fragments import JaxFragments, render_generated_script_from_fragments
    from local_agent.agent.session_spec import load_session_spec
    from lib.utils.source_stamp import write_stamp

    spec = load_session_spec(SESSION / "inputs" / "user_input.yaml")
    fragments = JaxFragments(
        rhs=("dose_rate - k_elim * D", "k_on * D * (1.0 - R) - k_off * R"),
        helper_functions=(),
        loss_body="""signal_model = baseline + gain * solution[:, 1]
signal = dataset[:, 1]
mask = jnp.isfinite(signal)
count = jnp.sum(mask)
scale = jnp.max(jnp.abs(jnp.where(mask, signal, 0.0)))
residuals = jnp.where(mask, (signal_model - signal) / scale, 0.0)
return jnp.where(count > 0, jnp.sqrt(jnp.sum(residuals * residuals) / count), jnp.nan)""",
        writeout_body="""signal_model = baseline + gain * solution[:, 1]
writeout_array = np.zeros([solution_time.shape[0], 5])
writeout_array[:, 0] = np.asarray(solution_time)
writeout_array[:, 1] = np.asarray(solution[:, 0])
writeout_array[:, 2] = np.asarray(solution[:, 1])
writeout_array[:, 3] = np.asarray(dataset[:, 1])
writeout_array[:, 4] = np.asarray(signal_model)
return writeout_array""",
    )
    (SESSION / "generated" / "generated_script.py").write_text(render_generated_script_from_fragments(fragments, spec))
    write_stamp(SESSION)


def main() -> None:
    make_inputs()
    write_user_model()
    write_generated_script()
    print(SESSION)


if __name__ == "__main__":
    main()

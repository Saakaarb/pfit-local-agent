# pfit-sources: user_model.py=10701b7ef77f85b3 user_input.yaml=db093f4c46f8056b
import jax
import jax.numpy as jnp
import numpy as np
import diffrax
from diffrax import RESULTS

jax.config.update("jax_enable_x64", True)

@jax.jit
def unscale_value(val, min_val, max_val, is_logscale):
    lin_unscaled = ((val + 1.0) / 2.0) * (max_val - min_val) + min_val
    unscaled = jnp.where(is_logscale, 10.0**lin_unscaled, lin_unscaled)
    return unscaled



@jax.jit
def user_defined_system(t, y, other_args):
    constants = other_args["constants"]
    trainable_variables = other_args["trainable_variables"]
    dataset = constants["dataset"]
    t_eval = constants["t_eval"]
    unscaled_parameters = unscale_value(trainable_variables, constants["min_limits"], constants["max_limits"], constants["is_logscale"])
    k_elim, k_on, k_off, baseline, gain = unscaled_parameters
    trainable_parameters = {"k_elim": k_elim, "k_on": k_on, "k_off": k_off, "baseline": baseline, "gain": gain}
    fixed_parameters = constants["fixed_parameters"]
    D = y[0]
    R = y[1]
    dose_rate = jnp.interp(t, t_eval, dataset[:, 0])
    dDdt = dose_rate - k_elim * D
    dRdt = k_on * D * (1.0 - R) - k_off * R
    return jnp.array([dDdt, dRdt])

@jax.jit
def _integrate_system_with_stats(constants, trainable_variables):
    term = diffrax.ODETerm(user_defined_system)
    solver = diffrax.Tsit5()
    t_eval = constants["t_eval"]
    other_args = {"constants": constants, "trainable_variables": trainable_variables}
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=constants["init_time"],
        t1=t_eval[-1],
        max_steps=10000,
        dt0=constants["init_timestep"],
        y0=constants["init_cond"],
        args=other_args,
        saveat=diffrax.SaveAt(ts=t_eval),
        throw=False,
        stepsize_controller=diffrax.PIDController(
            rtol=constants["stepsize_rtol"],
            atol=constants["stepsize_atol"],
        ),
    )
    return sol.ts, sol.ys, sol.result, sol.stats

@jax.jit
def _integrate_system(constants, trainable_variables):
    ts, ys, result, _ = _integrate_system_with_stats(constants, trainable_variables)
    return ts, ys, result

@jax.jit
def _compute_loss_value(constants, trainable_variables, solution_time, solution):
    dataset = constants["dataset"]
    unscaled_parameters = unscale_value(trainable_variables, constants["min_limits"], constants["max_limits"], constants["is_logscale"])
    k_elim, k_on, k_off, baseline, gain = unscaled_parameters
    trainable_parameters = {"k_elim": k_elim, "k_on": k_on, "k_off": k_off, "baseline": baseline, "gain": gain}
    fixed_parameters = constants["fixed_parameters"]
    signal_model = baseline + gain * solution[:, 1]
    signal = dataset[:, 1]
    mask = jnp.isfinite(signal)
    count = jnp.sum(mask)
    scale = jnp.max(jnp.abs(jnp.where(mask, signal, 0.0)))
    residuals = jnp.where(mask, (signal_model - signal) / scale, 0.0)
    return jnp.where(count > 0, jnp.sqrt(jnp.sum(residuals * residuals) / count), jnp.nan)

@jax.jit
def _compute_loss_problem(constants, trainable_variables):
    solution_time, solution, result = _integrate_system(constants, trainable_variables)
    failed = result != RESULTS.successful
    loss_value = _compute_loss_value(constants, trainable_variables, solution_time, solution)
    failed = failed | ~jnp.all(jnp.isfinite(solution)) | ~jnp.isfinite(loss_value)
    return jnp.where(failed, constants["error_loss"], loss_value)

def _write_problem_result(constants, trainable_variables):
    dataset = constants["dataset"]
    solution_time, solution, result = _integrate_system(constants, trainable_variables)
    unscaled_parameters = unscale_value(trainable_variables, constants["min_limits"], constants["max_limits"], constants["is_logscale"])
    k_elim, k_on, k_off, baseline, gain = unscaled_parameters
    trainable_parameters = {"k_elim": k_elim, "k_on": k_on, "k_off": k_off, "baseline": baseline, "gain": gain}
    fixed_parameters = constants["fixed_parameters"]
    signal_model = baseline + gain * solution[:, 1]
    writeout_array = np.zeros([solution_time.shape[0], 5])
    writeout_array[:, 0] = np.asarray(solution_time)
    writeout_array[:, 1] = np.asarray(solution[:, 0])
    writeout_array[:, 2] = np.asarray(solution[:, 1])
    writeout_array[:, 3] = np.asarray(dataset[:, 1])
    writeout_array[:, 4] = np.asarray(signal_model)
    return writeout_array

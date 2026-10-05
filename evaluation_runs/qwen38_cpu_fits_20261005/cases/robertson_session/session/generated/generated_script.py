# pfit-sources: user_model.py=e9ba374e1a531f16 user_input.yaml=580f0f66957bb87b
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
    k1, k2, k3 = unscaled_parameters
    trainable_parameters = {"k1": k1, "k2": k2, "k3": k3}
    fixed_parameters = constants["fixed_parameters"]
    y1 = y[0]
    y2 = y[1]
    y3 = y[2]

    dy1dt = -k1 * y1 + k3 * y3 * y2
    dy2dt = k1 * y1 - k2 * y2 ** 2 - k3 * y2 * y3
    dy3dt = k2 * y2 ** 2
    return jnp.array([dy1dt, dy2dt, dy3dt])

@jax.jit
def _integrate_system_with_stats(constants, trainable_variables):
    term = diffrax.ODETerm(user_defined_system)
    solver = diffrax.Kvaerno5()
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
    k1, k2, k3 = unscaled_parameters
    trainable_parameters = {"k1": k1, "k2": k2, "k3": k3}
    fixed_parameters = constants["fixed_parameters"]
    loss = 0.0
    loss += jnp.mean(jnp.square((solution[:, 0] - dataset[:, 0]) / (jnp.max(jnp.abs(dataset[:, 0])) + 1e-12)))
    loss += jnp.mean(jnp.square((solution[:, 1] - dataset[:, 1]) / (jnp.max(jnp.abs(dataset[:, 1])) + 1e-12)))
    loss += jnp.mean(jnp.square((solution[:, 2] - dataset[:, 2]) / (jnp.max(jnp.abs(dataset[:, 2])) + 1e-12)))
    loss = jnp.sqrt(loss / 3)
    return loss

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
    k1, k2, k3 = unscaled_parameters
    trainable_parameters = {"k1": k1, "k2": k2, "k3": k3}
    fixed_parameters = constants["fixed_parameters"]
    writeout_array = np.zeros([solution_time.shape[0], 7])
    writeout_array[:, 0] = solution_time
    writeout_array[:, 1] = dataset[:, 0]
    writeout_array[:, 2] = dataset[:, 1]
    writeout_array[:, 3] = dataset[:, 2]
    writeout_array[:, 4] = solution[:, 0]
    writeout_array[:, 5] = solution[:, 1]
    writeout_array[:, 6] = solution[:, 2]
    return writeout_array

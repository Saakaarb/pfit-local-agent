import jax
import jax.numpy as jnp
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
    unused_constant = fixed_parameters['unused_constant']
    y1 = y[0]
    y2 = y[1]
    y3 = y[2]
    dy1dt = -k1 * y1 + k3 * y3 * y2
    dy2dt = k1 * y1 - k2 * y2 ** 2 - k3 * y2 * y3
    dy3dt = k2 * y2 ** 2
    return jnp.array([dy1dt, dy2dt, dy3dt])

@jax.jit
def _integrate_system(constants, trainable_variables):
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
    return sol.ts, sol.ys, sol.result

@jax.jit
def _compute_loss_value(constants, trainable_variables, solution_time, solution):
    dataset = constants["dataset"]
    unscaled_parameters = unscale_value(trainable_variables, constants["min_limits"], constants["max_limits"], constants["is_logscale"])
    k1, k2, k3 = unscaled_parameters
    trainable_parameters = {"k1": k1, "k2": k2, "k3": k3}
    fixed_parameters = constants["fixed_parameters"]
    unused_constant = fixed_parameters['unused_constant']
    measured = jnp.stack([dataset[:, 0], dataset[:, 1], dataset[:, 2]], axis=1)
    simulated = jnp.stack([solution[:, 0], solution[:, 1], solution[:, 2]], axis=1)
    scale = jnp.maximum(jnp.max(jnp.abs(measured), axis=0), 1e-12)
    mask = jnp.isfinite(measured) & jnp.isfinite(simulated)
    measured_safe = jnp.where(mask, measured, 0.0)
    simulated_safe = jnp.where(mask, simulated, 0.0)
    residuals = jnp.where(mask, (simulated_safe - measured_safe) / scale, 0.0)
    count = jnp.maximum(jnp.sum(mask), 1)
    return jnp.sqrt(jnp.sum(residuals * residuals) / count)

@jax.jit
def _compute_loss_problem(constants, trainable_variables):
    solution_time, solution, result = _integrate_system(constants, trainable_variables)
    failed = jnp.logical_or(result == RESULTS.max_steps_reached, result == RESULTS.singular)
    loss_value = _compute_loss_value(constants, trainable_variables, solution_time, solution)
    return jnp.where(failed, constants["error_loss"], loss_value)

def _write_problem_result(constants, trainable_variables):
    dataset = constants["dataset"]
    solution_time, solution, result = _integrate_system(constants, trainable_variables)
    unscaled_parameters = unscale_value(trainable_variables, constants["min_limits"], constants["max_limits"], constants["is_logscale"])
    k1, k2, k3 = unscaled_parameters
    trainable_parameters = {"k1": k1, "k2": k2, "k3": k3}
    fixed_parameters = constants["fixed_parameters"]
    unused_constant = fixed_parameters['unused_constant']
    return jnp.column_stack((solution_time, dataset[:, 0], dataset[:, 1], dataset[:, 2], solution[:, 0], solution[:, 1], solution[:, 2]))

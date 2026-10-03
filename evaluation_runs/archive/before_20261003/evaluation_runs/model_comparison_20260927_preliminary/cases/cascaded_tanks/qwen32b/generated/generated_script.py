# pfit-sources: user_model.py=67f1148b849d5e66 user_input.yaml=c1df5e307e1368e1
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

def _observables(solution, k1, k2, k3, k4):
    x1 = solution[:, 0]
    x2 = solution[:, 1]
    return {'level_lower': x2}

@jax.jit
def user_defined_system(t, y, other_args):
    constants = other_args["constants"]
    trainable_variables = other_args["trainable_variables"]
    dataset = constants["dataset"]
    t_eval = constants["t_eval"]
    unscaled_parameters = unscale_value(trainable_variables, constants["min_limits"], constants["max_limits"], constants["is_logscale"])
    k1, k2, k3, k4 = unscaled_parameters
    trainable_parameters = {"k1": k1, "k2": k2, "k3": k3, "k4": k4}
    fixed_parameters = constants["fixed_parameters"]
    x1 = y[0]
    x2 = y[1]
    pump_voltage = jnp.interp(t, t_eval, dataset[:, 1])
    dx1dt = -k1 * jnp.sqrt(jnp.maximum(x1, 0)) + k4 * jnp.interp(t, t_eval, dataset[:, 1])
    dx2dt = k2 * jnp.sqrt(jnp.maximum(x1, 0)) - k3 * jnp.sqrt(jnp.maximum(x2, 0))
    return jnp.array([dx1dt, dx2dt])

@jax.jit
def _integrate_system(constants, trainable_variables):
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
    return sol.ts, sol.ys, sol.result

@jax.jit
def _compute_loss_value(constants, trainable_variables, solution_time, solution):
    dataset = constants["dataset"]
    unscaled_parameters = unscale_value(trainable_variables, constants["min_limits"], constants["max_limits"], constants["is_logscale"])
    k1, k2, k3, k4 = unscaled_parameters
    trainable_parameters = {"k1": k1, "k2": k2, "k3": k3, "k4": k4}
    fixed_parameters = constants["fixed_parameters"]
    k1 = trainable_parameters['k1']
    k2 = trainable_parameters['k2']
    k3 = trainable_parameters['k3']
    k4 = trainable_parameters['k4']
    observables = _observables(solution, k1, k2, k3, k4)
    loss = 0.0
    loss += jnp.mean(jnp.square((solution[:, 1] - dataset[:, 0]) / (jnp.max(jnp.abs(dataset[:, 0])) + 1e-12)))
    loss = jnp.sqrt(loss / 1)
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
    k1, k2, k3, k4 = unscaled_parameters
    trainable_parameters = {"k1": k1, "k2": k2, "k3": k3, "k4": k4}
    fixed_parameters = constants["fixed_parameters"]
    k1 = trainable_parameters['k1']
    k2 = trainable_parameters['k2']
    k3 = trainable_parameters['k3']
    k4 = trainable_parameters['k4']
    writeout_array = jnp.zeros([solution_time.shape[0], 5])
    writeout_array = writeout_array.at[:, 0].set(solution_time)
    writeout_array = writeout_array.at[:, 1].set(dataset[:, 0])
    writeout_array = writeout_array.at[:, 2].set(solution[:, 0])
    writeout_array = writeout_array.at[:, 3].set(solution[:, 1])
    observables = _observables(solution, k1, k2, k3, k4)
    writeout_array = writeout_array.at[:, 4].set(observables['level_lower'])
    return writeout_array

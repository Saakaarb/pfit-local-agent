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

def _observables(solution, trainable_parameters, fixed_parameters):
    ka = trainable_parameters['ka']
    ke = trainable_parameters['ke']
    Vd = trainable_parameters['Vd']
    A_gut = solution[:, 0]
    A_plasma = solution[:, 1]
    return {'C_plasma': A_plasma / Vd}

@jax.jit
def user_defined_system(t, y, other_args):
    constants = other_args["constants"]
    trainable_variables = other_args["trainable_variables"]
    dataset = constants["dataset"]
    t_eval = constants["t_eval"]
    unscaled_parameters = unscale_value(trainable_variables, constants["min_limits"], constants["max_limits"], constants["is_logscale"])
    ka, ke, Vd = unscaled_parameters
    trainable_parameters = {"ka": ka, "ke": ke, "Vd": Vd}
    fixed_parameters = constants["fixed_parameters"]
    A_gut = y[0]
    A_plasma = y[1]
    dA_gutdt = -ka * A_gut
    dA_plasmadt = ka * A_gut - ke * A_plasma
    return jnp.array([dA_gutdt, dA_plasmadt])

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
    ka, ke, Vd = unscaled_parameters
    trainable_parameters = {"ka": ka, "ke": ke, "Vd": Vd}
    fixed_parameters = constants["fixed_parameters"]
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    loss = 0.0
    loss += jnp.mean(jnp.square((observables['C_plasma'] - dataset[:, 0]) / (jnp.max(dataset[:, 0]) - jnp.min(dataset[:, 0]) + 1e-12)))
    return loss

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
    ka, ke, Vd = unscaled_parameters
    trainable_parameters = {"ka": ka, "ke": ke, "Vd": Vd}
    fixed_parameters = constants["fixed_parameters"]
    Nts = solution_time.shape[0]
    out = jnp.zeros((Nts, 5))
    out = out.at[:, 0].set(solution_time)
    out = out.at[:, 1].set(dataset[:, 0])
    out = out.at[:, 2].set(solution[:, 0])
    out = out.at[:, 3].set(solution[:, 1])
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    out = out.at[:, 4].set(observables['C_plasma'])
    return out

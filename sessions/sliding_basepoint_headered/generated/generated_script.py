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
    c2 = trainable_parameters['c2']
    Dk = trainable_parameters['Dk']
    Dc = trainable_parameters['Dc']
    m1 = trainable_parameters['m1']
    m2 = trainable_parameters['m2']
    vf = fixed_parameters['vf']
    x1 = solution[:, 0]
    x2 = solution[:, 1]
    v1 = solution[:, 2]
    v2 = solution[:, 3]
    k = solution[:, 4]
    c1 = solution[:, 5]
    return {'contact_force': jnp.abs(k * (x2 - x1) - c1 * jnp.abs(v1) * jnp.sign(v1)) / 1000, 'displacement': x1}

@jax.jit
def user_defined_system(t, y, other_args):
    constants = other_args["constants"]
    trainable_variables = other_args["trainable_variables"]
    dataset = constants["dataset"]
    t_eval = constants["t_eval"]
    unscaled_parameters = unscale_value(trainable_variables, constants["min_limits"], constants["max_limits"], constants["is_logscale"])
    c2, Dk, Dc, m1, m2 = unscaled_parameters
    trainable_parameters = {"c2": c2, "Dk": Dk, "Dc": Dc, "m1": m1, "m2": m2}
    fixed_parameters = constants["fixed_parameters"]
    vf = fixed_parameters['vf']
    x1 = y[0]
    x2 = y[1]
    v1 = y[2]
    v2 = y[3]
    k = y[4]
    c1 = y[5]
    dx1dt = v1
    dx2dt = v2
    dv1dt = (k * (x2 - x1) - c1 * jnp.abs(v1) * jnp.sign(v1)) / m1
    dv2dt = 0.5 * (1 + jnp.tanh((x2 - x1) / 0.01)) * (-k * (x2 - x1) - c2 * jnp.sign(v2)) / m2
    dkdt = Dk * jnp.abs(m1 * v1 * (k * (x2 - x1) - c1 * jnp.abs(v1) * jnp.sign(v1)) / m1)
    dc1dt = Dc * jnp.abs(m1 * v1 * (k * (x2 - x1) - c1 * jnp.abs(v1) * jnp.sign(v1)) / m1)
    return jnp.array([dx1dt, dx2dt, dv1dt, dv2dt, dkdt, dc1dt])

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
    c2, Dk, Dc, m1, m2 = unscaled_parameters
    trainable_parameters = {"c2": c2, "Dk": Dk, "Dc": Dc, "m1": m1, "m2": m2}
    fixed_parameters = constants["fixed_parameters"]
    vf = fixed_parameters['vf']
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    loss = 0.0
    loss += jnp.mean(jnp.square((observables['contact_force'] - dataset[:, 0]) / (jnp.max(dataset[:, 0]) - jnp.min(dataset[:, 0]) + 1e-12)))
    loss += jnp.mean(jnp.square((observables['displacement'] - dataset[:, 1]) / (jnp.max(dataset[:, 1]) - jnp.min(dataset[:, 1]) + 1e-12)))
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
    c2, Dk, Dc, m1, m2 = unscaled_parameters
    trainable_parameters = {"c2": c2, "Dk": Dk, "Dc": Dc, "m1": m1, "m2": m2}
    fixed_parameters = constants["fixed_parameters"]
    vf = fixed_parameters['vf']
    Nts = solution_time.shape[0]
    out = jnp.zeros((Nts, 11))
    out = out.at[:, 0].set(solution_time)
    out = out.at[:, 1].set(dataset[:, 0])
    out = out.at[:, 2].set(dataset[:, 1])
    out = out.at[:, 3].set(solution[:, 0])
    out = out.at[:, 4].set(solution[:, 1])
    out = out.at[:, 5].set(solution[:, 2])
    out = out.at[:, 6].set(solution[:, 3])
    out = out.at[:, 7].set(solution[:, 4])
    out = out.at[:, 8].set(solution[:, 5])
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    out = out.at[:, 9].set(observables['contact_force'])
    out = out.at[:, 10].set(observables['displacement'])
    return out

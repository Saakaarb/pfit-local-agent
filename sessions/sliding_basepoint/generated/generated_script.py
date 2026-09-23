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
    x1 = solution[:, 0]
    x2 = solution[:, 1]
    v1 = solution[:, 2]
    k = solution[:, 4]
    c1 = solution[:, 5]
    Fs = k * (x2 - x1)
    return {'contact_force': jnp.abs(Fs - c1 * jnp.abs(v1) * jnp.sign(v1))}

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
    dv2dt = jnp.where(jnp.logical_and(jnp.abs(-k * (x2 - x1)) < c2, jnp.abs(v2) < vf), 0.0, -k * (x2 - x1) - c2 * jnp.sign(v2)) / m2
    dkdt = Dk * jnp.abs(m1 * v1 * (k * (x2 - x1) - c1 * jnp.abs(v1) * jnp.sign(v1)) / m1)
    dc1dt = Dc * jnp.abs(m1 * v1 * (k * (x2 - x1) - c1 * jnp.abs(v1) * jnp.sign(v1)) / m1)
    return jnp.array([dx1dt, dx2dt, dv1dt, dv2dt, dkdt, dc1dt])

@jax.jit
def _integrate_system(constants, trainable_variables):
    term = diffrax.ODETerm(user_defined_system)
    solver = diffrax.Dopri8()
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
    F_sim = _observables(solution, trainable_parameters, fixed_parameters)['contact_force']
    s_sim = solution[:, 0]
    F_exp = 1000.0 * dataset[:, 0]
    s_exp = dataset[:, 1]
    loss_F = jnp.mean(jnp.abs(F_exp - F_sim)) / jnp.max(jnp.abs(F_exp))
    loss_s = jnp.mean(jnp.abs(s_exp - s_sim)) / jnp.max(jnp.abs(s_exp))
    return loss_F + loss_s

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
    F_sim = _observables(solution, trainable_parameters, fixed_parameters)['contact_force']
    s_sim = solution[:, 0]

    F_exp = 1000.0 * dataset[:, 0]
    s_exp = dataset[:, 1]

    Nts = solution_time.shape[0]
    writeout_array = jnp.zeros((Nts, 5))
    writeout_array = writeout_array.at[:, 0].set(solution_time)
    writeout_array = writeout_array.at[:, 1].set(F_exp)
    writeout_array = writeout_array.at[:, 2].set(s_exp)
    writeout_array = writeout_array.at[:, 3].set(F_sim)
    writeout_array = writeout_array.at[:, 4].set(s_sim)
    return writeout_array

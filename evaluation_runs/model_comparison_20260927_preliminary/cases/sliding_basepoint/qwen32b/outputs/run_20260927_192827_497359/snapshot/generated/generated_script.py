# pfit-sources: user_model.py=10c84801a718842c user_input.yaml=66f8c3bd5cd94e4d
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

def _observables(solution, c2, Dk, Dc, m1, m2, vf):
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
    dv2dt = jnp.where(jnp.logical_and(jnp.abs(-k * (x2 - x1)) < c2, jnp.abs(v2) < vf), 0, (-k * (x2 - x1) - c2 * jnp.sign(v2)) / m2)
    dkdt = Dk * jnp.abs(m1 * v1 * ((k * (x2 - x1) - c1 * jnp.abs(v1) * jnp.sign(v1)) / m1))
    dc1dt = Dc * jnp.abs(m1 * v1 * ((k * (x2 - x1) - c1 * jnp.abs(v1) * jnp.sign(v1)) / m1))
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
    observables = _observables(solution, trainable_parameters['c2'], trainable_parameters['Dk'], trainable_parameters['Dc'], trainable_parameters['m1'], trainable_parameters['m2'], fixed_parameters['vf'])
    loss = 0.0
    loss += jnp.mean(jnp.square((observables['contact_force'] - dataset[:, 0]) / (jnp.max(jnp.abs(dataset[:, 0])) + 1e-12)))
    loss += jnp.mean(jnp.square((observables['displacement'] - dataset[:, 1]) / (jnp.max(jnp.abs(dataset[:, 1])) + 1e-12)))
    loss = jnp.sqrt(loss / 2)
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
    c2, Dk, Dc, m1, m2 = unscaled_parameters
    trainable_parameters = {"c2": c2, "Dk": Dk, "Dc": Dc, "m1": m1, "m2": m2}
    fixed_parameters = constants["fixed_parameters"]
    vf = fixed_parameters['vf']
    writeout_array = np.zeros([solution_time.shape[0], 11])
    writeout_array[:, 0] = solution_time
    writeout_array[:, 1] = dataset[:, 0]
    writeout_array[:, 2] = dataset[:, 1]
    writeout_array[:, 3] = solution[:, 0]
    writeout_array[:, 4] = solution[:, 1]
    writeout_array[:, 5] = solution[:, 2]
    writeout_array[:, 6] = solution[:, 3]
    writeout_array[:, 7] = solution[:, 4]
    writeout_array[:, 8] = solution[:, 5]
    observables = _observables(solution, trainable_parameters['c2'], trainable_parameters['Dk'], trainable_parameters['Dc'], trainable_parameters['m1'], trainable_parameters['m2'], fixed_parameters['vf'])
    writeout_array[:, 9] = observables['contact_force']
    writeout_array[:, 10] = observables['displacement']
    return writeout_array

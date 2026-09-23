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
    Ea1 = trainable_parameters['Ea1']
    h1 = trainable_parameters['h1']
    A1 = trainable_parameters['A1']
    A2 = trainable_parameters['A2']
    Ea2 = trainable_parameters['Ea2']
    h2 = trainable_parameters['h2']
    m2 = trainable_parameters['m2']
    n2 = trainable_parameters['n2']
    kb = fixed_parameters['kb']
    c1 = solution[:, 0]
    c2 = solution[:, 1]
    T = solution[:, 2]
    return {'dTdt': jnp.abs(h1 * (-A1 * jnp.exp(-Ea1 / (kb * T)) * c1)) + jnp.abs(h2 * (A2 * jnp.exp(-Ea2 / (kb * T)) * c2 ** n2 * (1 - c2) ** m2))}

@jax.jit
def user_defined_system(t, y, other_args):
    constants = other_args["constants"]
    trainable_variables = other_args["trainable_variables"]
    dataset = constants["dataset"]
    t_eval = constants["t_eval"]
    unscaled_parameters = unscale_value(trainable_variables, constants["min_limits"], constants["max_limits"], constants["is_logscale"])
    Ea1, h1, A1, A2, Ea2, h2, m2, n2 = unscaled_parameters
    trainable_parameters = {"Ea1": Ea1, "h1": h1, "A1": A1, "A2": A2, "Ea2": Ea2, "h2": h2, "m2": m2, "n2": n2}
    fixed_parameters = constants["fixed_parameters"]
    kb = fixed_parameters['kb']
    c1 = y[0]
    c2 = y[1]
    T = y[2]
    dc1dt = -A1 * jnp.exp(-Ea1 / (kb * T)) * c1
    dc2dt = A2 * jnp.exp(-Ea2 / (kb * T)) * c2 ** n2 * (1 - c2) ** m2
    dTdt = jnp.abs(h1 * (-A1 * jnp.exp(-Ea1 / (kb * T)) * c1)) + jnp.abs(h2 * (A2 * jnp.exp(-Ea2 / (kb * T)) * c2 ** n2 * (1 - c2) ** m2))
    return jnp.array([dc1dt, dc2dt, dTdt])

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
    Ea1, h1, A1, A2, Ea2, h2, m2, n2 = unscaled_parameters
    trainable_parameters = {"Ea1": Ea1, "h1": h1, "A1": A1, "A2": A2, "Ea2": Ea2, "h2": h2, "m2": m2, "n2": n2}
    fixed_parameters = constants["fixed_parameters"]
    kb = fixed_parameters['kb']
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    loss = 0.0
    loss += jnp.mean(jnp.square((solution[:, 2] - dataset[:, 0]) / (jnp.max(dataset[:, 0]) - jnp.min(dataset[:, 0]) + 1e-12)))
    eps_dTdt = jnp.min(jnp.where(dataset[:, 1] > 0.0, dataset[:, 1], jnp.inf))
    log_sim_dTdt = jnp.log10(observables['dTdt'] + eps_dTdt)
    log_measured_dTdt = jnp.log10(dataset[:, 1] + eps_dTdt)
    scale_log_dTdt = jnp.max(log_measured_dTdt) - jnp.min(log_measured_dTdt) + 1e-12
    loss += jnp.mean(jnp.square((log_sim_dTdt - log_measured_dTdt) / scale_log_dTdt))
    loss += 10000.0 / (1.0 + jnp.exp(-jnp.clip(1000.0 * (solution[-1, 0] - 0.02), -60.0, 60.0)))
    loss += 10000.0 / (1.0 + jnp.exp(-jnp.clip(1000.0 * (0.98 - solution[-1, 1]), -60.0, 60.0)))
    loss += 10000.0 / (1.0 + jnp.exp(-jnp.clip(1000.0 * (jnp.sqrt(jnp.square(solution[-1, 2] - dataset[-1, 0]) + 1e-12) - 50.0), -60.0, 60.0)))
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
    Ea1, h1, A1, A2, Ea2, h2, m2, n2 = unscaled_parameters
    trainable_parameters = {"Ea1": Ea1, "h1": h1, "A1": A1, "A2": A2, "Ea2": Ea2, "h2": h2, "m2": m2, "n2": n2}
    fixed_parameters = constants["fixed_parameters"]
    kb = fixed_parameters['kb']
    Nts = solution_time.shape[0]
    out = jnp.zeros((Nts, 7))
    out = out.at[:, 0].set(solution_time)
    out = out.at[:, 1].set(dataset[:, 0])
    out = out.at[:, 2].set(dataset[:, 1])
    out = out.at[:, 3].set(solution[:, 0])
    out = out.at[:, 4].set(solution[:, 1])
    out = out.at[:, 5].set(solution[:, 2])
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    out = out.at[:, 6].set(observables['dTdt'])
    return out

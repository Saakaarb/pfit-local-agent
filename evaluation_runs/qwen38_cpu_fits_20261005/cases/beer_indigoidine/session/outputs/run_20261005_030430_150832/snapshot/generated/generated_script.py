# pfit-sources: user_model.py=033ac6579e6370e0 user_input.yaml=2beec46b2c9276ea
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
    Bacmax = trainable_parameters['Bacmax']
    beta = trainable_parameters['beta']
    ksyn = trainable_parameters['ksyn']
    kdim = trainable_parameters['kdim']
    kdegi = trainable_parameters['kdegi']
    tau = fixed_parameters['tau']
    Bac = solution[:, 0]
    Glu = solution[:, 1]
    cGlu = solution[:, 2]
    Ind = solution[:, 3]
    return {'Bacnorm': Bac, 'IndconcNormRange': Ind}

@jax.jit
def user_defined_system(t, y, other_args):
    constants = other_args["constants"]
    trainable_variables = other_args["trainable_variables"]
    dataset = constants["dataset"]
    t_eval = constants["t_eval"]
    unscaled_parameters = unscale_value(trainable_variables, constants["min_limits"], constants["max_limits"], constants["is_logscale"])
    Bacmax, beta, ksyn, kdim, kdegi = unscaled_parameters
    trainable_parameters = {"Bacmax": Bacmax, "beta": beta, "ksyn": ksyn, "kdim": kdim, "kdegi": kdegi}
    fixed_parameters = constants["fixed_parameters"]
    tau = fixed_parameters['tau']
    Bac = y[0]
    Glu = y[1]
    cGlu = y[2]
    Ind = y[3]

    dBacdt = beta * Bac * (0.5 * (1 + jnp.tanh((t - tau) / 0.01))) * (1 - Bac / Bacmax)
    dGludt = -ksyn * Bac * Glu
    dcGludt = ksyn * Bac * Glu - kdim * cGlu ** 2
    dInddt = kdim * cGlu ** 2 - kdegi * Ind
    return jnp.array([dBacdt, dGludt, dcGludt, dInddt])

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
    Bacmax, beta, ksyn, kdim, kdegi = unscaled_parameters
    trainable_parameters = {"Bacmax": Bacmax, "beta": beta, "ksyn": ksyn, "kdim": kdim, "kdegi": kdegi}
    fixed_parameters = constants["fixed_parameters"]
    tau = fixed_parameters['tau']
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    loss = 0.0
    loss += jnp.mean(jnp.square((solution[:, 0] - dataset[:, 0]) / (jnp.max(jnp.abs(dataset[:, 0])) + 1e-12)))
    loss += jnp.mean(jnp.square((solution[:, 3] - dataset[:, 1]) / (jnp.max(jnp.abs(dataset[:, 1])) + 1e-12)))
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
    Bacmax, beta, ksyn, kdim, kdegi = unscaled_parameters
    trainable_parameters = {"Bacmax": Bacmax, "beta": beta, "ksyn": ksyn, "kdim": kdim, "kdegi": kdegi}
    fixed_parameters = constants["fixed_parameters"]
    tau = fixed_parameters['tau']
    writeout_array = np.zeros([solution_time.shape[0], 9])
    writeout_array[:, 0] = solution_time
    writeout_array[:, 1] = dataset[:, 0]
    writeout_array[:, 2] = dataset[:, 1]
    writeout_array[:, 3] = solution[:, 0]
    writeout_array[:, 4] = solution[:, 1]
    writeout_array[:, 5] = solution[:, 2]
    writeout_array[:, 6] = solution[:, 3]
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    writeout_array[:, 7] = observables['Bacnorm']
    writeout_array[:, 8] = observables['IndconcNormRange']
    return writeout_array

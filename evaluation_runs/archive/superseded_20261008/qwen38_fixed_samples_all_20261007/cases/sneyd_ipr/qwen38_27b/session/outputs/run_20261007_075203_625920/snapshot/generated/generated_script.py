# pfit-sources: user_model.py=8d51853a6636350d user_input.yaml=9f3fc88fb3b27c88
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
    k1 = trainable_parameters['k1']
    k2 = trainable_parameters['k2']
    k3 = trainable_parameters['k3']
    k4 = trainable_parameters['k4']
    k_1 = trainable_parameters['k_1']
    k_2 = trainable_parameters['k_2']
    k_3 = trainable_parameters['k_3']
    k_4 = trainable_parameters['k_4']
    l2 = trainable_parameters['l2']
    l4 = trainable_parameters['l4']
    l6 = trainable_parameters['l6']
    l_2 = trainable_parameters['l_2']
    l_4 = trainable_parameters['l_4']
    l_6 = trainable_parameters['l_6']
    O = solution[:, 0]
    A = solution[:, 4]
    return {'open_probability': (0.9 * A + 0.1 * O) ** 4}

@jax.jit
def user_defined_system(t, y, other_args):
    constants = other_args["constants"]
    trainable_variables = other_args["trainable_variables"]
    dataset = constants["dataset"]
    t_eval = constants["t_eval"]
    unscaled_parameters = unscale_value(trainable_variables, constants["min_limits"], constants["max_limits"], constants["is_logscale"])
    k1, k2, k3, k4, k_1, k_2, k_3, k_4, l2, l4, l6, l_2, l_4, l_6 = unscaled_parameters
    trainable_parameters = {"k1": k1, "k2": k2, "k3": k3, "k4": k4, "k_1": k_1, "k_2": k_2, "k_3": k_3, "k_4": k_4, "l2": l2, "l4": l4, "l6": l6, "l_2": l_2, "l_4": l_4, "l_6": l_6}
    fixed_parameters = constants["fixed_parameters"]
    O = y[0]
    R = y[1]
    I1 = y[2]
    S = y[3]
    A = y[4]
    I2 = y[5]
    IP3 = y[6]
    Ca = y[7]

    dOdt = -((k_2 + l_4 * Ca) / (1 + Ca / (k_4 * l6 / (k4 * l_6))) * O) + (k2 * (k_2 * l4 / (k2 * l_4)) + l4 * Ca) / (k_2 * l4 / (k2 * l_4) + Ca * (1 + k_2 * l4 / (k2 * l_4) / (k_1 * l2 / (k1 * l_2)))) * IP3 * R - (k4 * (k_4 * l6 / (k4 * l_6)) + l6) * Ca / (k_4 * l6 / (k4 * l_6) + Ca) * O + k_1 * l2 / (k1 * l_2) * (k_4 + l_6) / (k_1 * l2 / (k1 * l_2) + Ca) * A - k3 * (k_4 * l6 / (k4 * l_6)) / (k_4 * l6 / (k4 * l_6) + Ca) * O + k_3 * S
    dRdt = (k_2 + l_4 * Ca) / (1 + Ca / (k_4 * l6 / (k4 * l_6))) * O - (k2 * (k_2 * l4 / (k2 * l_4)) + l4 * Ca) / (k_2 * l4 / (k2 * l_4) + Ca * (1 + k_2 * l4 / (k2 * l_4) / (k_1 * l2 / (k1 * l_2)))) * IP3 * R - (k1 * (k_1 * l2 / (k1 * l_2)) + l2) * Ca / (k_1 * l2 / (k1 * l_2) + Ca * (1 + k_1 * l2 / (k1 * l_2) / (k_2 * l4 / (k2 * l_4)))) * R + (k_1 + l_2) * I1
    dI1dt = (k1 * (k_1 * l2 / (k1 * l_2)) + l2) * Ca / (k_1 * l2 / (k1 * l_2) + Ca * (1 + k_1 * l2 / (k1 * l_2) / (k_2 * l4 / (k2 * l_4)))) * R - (k_1 + l_2) * I1
    dSdt = k3 * (k_4 * l6 / (k4 * l_6)) / (k_4 * l6 / (k4 * l_6) + Ca) * O - k_3 * S
    dAdt = (k4 * (k_4 * l6 / (k4 * l_6)) + l6) * Ca / (k_4 * l6 / (k4 * l_6) + Ca) * O - k_1 * l2 / (k1 * l_2) * (k_4 + l_6) / (k_1 * l2 / (k1 * l_2) + Ca) * A - (k1 * (k_1 * l2 / (k1 * l_2)) + l2) * Ca / (k_1 * l2 / (k1 * l_2) + Ca) * A + (k_1 + l_2) * I2
    dI2dt = (k1 * (k_1 * l2 / (k1 * l_2)) + l2) * Ca / (k_1 * l2 / (k1 * l_2) + Ca) * A - (k_1 + l_2) * I2
    dIP3dt = 0.0
    dCadt = 0.0
    return jnp.array([dOdt, dRdt, dI1dt, dSdt, dAdt, dI2dt, dIP3dt, dCadt])

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
        max_steps=1000,
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
    k1, k2, k3, k4, k_1, k_2, k_3, k_4, l2, l4, l6, l_2, l_4, l_6 = unscaled_parameters
    trainable_parameters = {"k1": k1, "k2": k2, "k3": k3, "k4": k4, "k_1": k_1, "k_2": k_2, "k_3": k_3, "k_4": k_4, "l2": l2, "l4": l4, "l6": l6, "l_2": l_2, "l_4": l_4, "l_6": l_6}
    fixed_parameters = constants["fixed_parameters"]
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    loss = 0.0
    loss += jnp.mean(jnp.square(observables['open_probability'] - dataset[:, 0]))
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
    k1, k2, k3, k4, k_1, k_2, k_3, k_4, l2, l4, l6, l_2, l_4, l_6 = unscaled_parameters
    trainable_parameters = {"k1": k1, "k2": k2, "k3": k3, "k4": k4, "k_1": k_1, "k_2": k_2, "k_3": k_3, "k_4": k_4, "l2": l2, "l4": l4, "l6": l6, "l_2": l_2, "l_4": l_4, "l_6": l_6}
    fixed_parameters = constants["fixed_parameters"]
    writeout_array = np.zeros([solution_time.shape[0], 11])
    writeout_array[:, 0] = solution_time
    writeout_array[:, 1] = dataset[:, 0]
    writeout_array[:, 2] = solution[:, 0]
    writeout_array[:, 3] = solution[:, 1]
    writeout_array[:, 4] = solution[:, 2]
    writeout_array[:, 5] = solution[:, 3]
    writeout_array[:, 6] = solution[:, 4]
    writeout_array[:, 7] = solution[:, 5]
    writeout_array[:, 8] = solution[:, 6]
    writeout_array[:, 9] = solution[:, 7]
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    writeout_array[:, 10] = observables['open_probability']
    return writeout_array

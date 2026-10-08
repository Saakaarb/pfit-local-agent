# pfit-sources: user_model.py=9a4fb2dfad62a1a5 user_input.yaml=5d4173521292c6a1
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
    offset = trainable_parameters['offset']
    scale = trainable_parameters['scale']
    Z = solution[:, 0]
    return {'Ca': offset + scale * Z}

@jax.jit
def user_defined_system(t, y, other_args):
    constants = other_args["constants"]
    trainable_variables = other_args["trainable_variables"]
    dataset = constants["dataset"]
    t_eval = constants["t_eval"]
    unscaled_parameters = unscale_value(trainable_variables, constants["min_limits"], constants["max_limits"], constants["is_logscale"])
    K2, K_par, Ka, Kd, Kf, Kp, Ky, Kz, Vd, Vm2, Vm3, Vp, beta_par, epsilon_par, n_par, v0, v1, offset, scale = unscaled_parameters
    trainable_parameters = {"K2": K2, "K_par": K_par, "Ka": Ka, "Kd": Kd, "Kf": Kf, "Kp": Kp, "Ky": Ky, "Kz": Kz, "Vd": Vd, "Vm2": Vm2, "Vm3": Vm3, "Vp": Vp, "beta_par": beta_par, "epsilon_par": epsilon_par, "n_par": n_par, "v0": v0, "v1": v1, "offset": offset, "scale": scale}
    fixed_parameters = constants["fixed_parameters"]
    Z = y[0]
    Y = y[1]
    A = y[2]

    dZdt = v0 + beta_par * v1 - Vm2 * Z ** 2 / (K2 ** 2 + Z ** 2) + Vm3 * A ** 4 * Y ** 2 * Z ** 4 / ((A ** 4 + Ka ** 4) * (Ky ** 2 + Y ** 2) * (Kz ** 4 + Z ** 4)) + Kf * Y - K_par * Z
    dYdt = Vm2 * Z ** 2 / (K2 ** 2 + Z ** 2) - Vm3 * A ** 4 * Y ** 2 * Z ** 4 / ((A ** 4 + Ka ** 4) * (Ky ** 2 + Y ** 2) * (Kz ** 4 + Z ** 4)) - Kf * Y
    dAdt = Vp * beta_par - Vd * A ** 2 * Z ** n_par / ((Kd ** n_par + Z ** n_par) * (A ** 2 + Kp ** 2)) - epsilon_par * A
    return jnp.array([dZdt, dYdt, dAdt])

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
    K2, K_par, Ka, Kd, Kf, Kp, Ky, Kz, Vd, Vm2, Vm3, Vp, beta_par, epsilon_par, n_par, v0, v1, offset, scale = unscaled_parameters
    trainable_parameters = {"K2": K2, "K_par": K_par, "Ka": Ka, "Kd": Kd, "Kf": Kf, "Kp": Kp, "Ky": Ky, "Kz": Kz, "Vd": Vd, "Vm2": Vm2, "Vm3": Vm3, "Vp": Vp, "beta_par": beta_par, "epsilon_par": epsilon_par, "n_par": n_par, "v0": v0, "v1": v1, "offset": offset, "scale": scale}
    fixed_parameters = constants["fixed_parameters"]
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    loss = 0.0
    loss += jnp.mean(jnp.square((solution[:, 0] - dataset[:, 0]) / (jnp.max(jnp.abs(dataset[:, 0])) + 1e-12)))
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
    K2, K_par, Ka, Kd, Kf, Kp, Ky, Kz, Vd, Vm2, Vm3, Vp, beta_par, epsilon_par, n_par, v0, v1, offset, scale = unscaled_parameters
    trainable_parameters = {"K2": K2, "K_par": K_par, "Ka": Ka, "Kd": Kd, "Kf": Kf, "Kp": Kp, "Ky": Ky, "Kz": Kz, "Vd": Vd, "Vm2": Vm2, "Vm3": Vm3, "Vp": Vp, "beta_par": beta_par, "epsilon_par": epsilon_par, "n_par": n_par, "v0": v0, "v1": v1, "offset": offset, "scale": scale}
    fixed_parameters = constants["fixed_parameters"]
    writeout_array = np.zeros([solution_time.shape[0], 6])
    writeout_array[:, 0] = solution_time
    writeout_array[:, 1] = dataset[:, 0]
    writeout_array[:, 2] = solution[:, 0]
    writeout_array[:, 3] = solution[:, 1]
    writeout_array[:, 4] = solution[:, 2]
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    writeout_array[:, 5] = observables['Ca']
    return writeout_array

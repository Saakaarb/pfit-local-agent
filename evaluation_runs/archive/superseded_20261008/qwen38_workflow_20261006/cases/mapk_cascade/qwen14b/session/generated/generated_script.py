# pfit-sources: user_model.py=9e9bab85fd162ea7 user_input.yaml=6aafbba374a846ca
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



@jax.jit
def user_defined_system(t, y, other_args):
    constants = other_args["constants"]
    trainable_variables = other_args["trainable_variables"]
    dataset = constants["dataset"]
    t_eval = constants["t_eval"]
    unscaled_parameters = unscale_value(trainable_variables, constants["min_limits"], constants["max_limits"], constants["is_logscale"])
    k1, k_1, k2, k3, k_3, k4, h1, h_1, h2, h3, h_3, h4, h_4, h5, h6, h_6 = unscaled_parameters
    trainable_parameters = {"k1": k1, "k_1": k_1, "k2": k2, "k3": k3, "k_3": k_3, "k4": k4, "h1": h1, "h_1": h_1, "h2": h2, "h3": h3, "h_3": h_3, "h4": h4, "h_4": h_4, "h5": h5, "h6": h6, "h_6": h_6}
    fixed_parameters = constants["fixed_parameters"]
    M = y[0]
    Mp = y[1]
    Mpp = y[2]
    MAPKK = y[3]
    MKP3 = y[4]
    M_MAPKK = y[5]
    Mp_MAPKK = y[6]
    Mpp_MKP3 = y[7]
    Mp_MKP3_dep = y[8]
    Mp_MKP3 = y[9]
    M_MKP3 = y[10]

    dMdt = -(k1 * M * MAPKK - k_1 * M_MAPKK) + (h6 * M_MKP3 - h_6 * M * MKP3)
    dMpdt = k2 * M_MAPKK - (k3 * Mp * MAPKK - k_3 * Mp_MAPKK) + (h3 * Mp_MKP3_dep - h_3 * Mp * MKP3) - (h4 * Mp * MKP3 - h_4 * Mp_MKP3)
    dMppdt = k4 * Mp_MAPKK - (h1 * Mpp * MKP3 - h_1 * Mpp_MKP3)
    dMAPKKdt = -(k1 * M * MAPKK - k_1 * M_MAPKK) + k2 * M_MAPKK - (k3 * Mp * MAPKK - k_3 * Mp_MAPKK) + k4 * Mp_MAPKK
    dMKP3dt = -(h1 * Mpp * MKP3 - h_1 * Mpp_MKP3) + (h3 * Mp_MKP3_dep - h_3 * Mp * MKP3) - (h4 * Mp * MKP3 - h_4 * Mp_MKP3) + (h6 * M_MKP3 - h_6 * M * MKP3)
    dM_MAPKKdt = k1 * M * MAPKK - k_1 * M_MAPKK - k2 * M_MAPKK
    dMp_MAPKKdt = k3 * Mp * MAPKK - k_3 * Mp_MAPKK - k4 * Mp_MAPKK
    dMpp_MKP3dt = h1 * Mpp * MKP3 - h_1 * Mpp_MKP3 - h2 * Mpp_MKP3
    dMp_MKP3_depdt = h2 * Mpp_MKP3 - (h3 * Mp_MKP3_dep - h_3 * Mp * MKP3)
    dMp_MKP3dt = h4 * Mp * MKP3 - h_4 * Mp_MKP3 - h5 * Mp_MKP3
    dM_MKP3dt = h5 * Mp_MKP3 - (h6 * M_MKP3 - h_6 * M * MKP3)
    return jnp.array([dMdt, dMpdt, dMppdt, dMAPKKdt, dMKP3dt, dM_MAPKKdt, dMp_MAPKKdt, dMpp_MKP3dt, dMp_MKP3_depdt, dMp_MKP3dt, dM_MKP3dt])

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
    k1, k_1, k2, k3, k_3, k4, h1, h_1, h2, h3, h_3, h4, h_4, h5, h6, h_6 = unscaled_parameters
    trainable_parameters = {"k1": k1, "k_1": k_1, "k2": k2, "k3": k3, "k_3": k_3, "k4": k4, "h1": h1, "h_1": h_1, "h2": h2, "h3": h3, "h_3": h_3, "h4": h4, "h_4": h_4, "h5": h5, "h6": h6, "h_6": h_6}
    fixed_parameters = constants["fixed_parameters"]
    loss = 0.0
    loss += jnp.mean(jnp.square((solution[:, 2] - dataset[:, 0]) / (jnp.max(jnp.abs(dataset[:, 0])) + 1e-12)))
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
    k1, k_1, k2, k3, k_3, k4, h1, h_1, h2, h3, h_3, h4, h_4, h5, h6, h_6 = unscaled_parameters
    trainable_parameters = {"k1": k1, "k_1": k_1, "k2": k2, "k3": k3, "k_3": k_3, "k4": k4, "h1": h1, "h_1": h_1, "h2": h2, "h3": h3, "h_3": h_3, "h4": h4, "h_4": h_4, "h5": h5, "h6": h6, "h_6": h_6}
    fixed_parameters = constants["fixed_parameters"]
    writeout_array = np.zeros([solution_time.shape[0], 13])
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
    writeout_array[:, 10] = solution[:, 8]
    writeout_array[:, 11] = solution[:, 9]
    writeout_array[:, 12] = solution[:, 10]
    return writeout_array

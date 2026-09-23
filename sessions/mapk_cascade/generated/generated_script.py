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
    k1, k_1, k2, k3, k_3, k4, h1, h_1, h2, h3, h_3, h4, h_4, h5, h6, h_6 = unscaled_parameters
    trainable_parameters = {"k1": k1, "k_1": k_1, "k2": k2, "k3": k3, "k_3": k_3, "k4": k4, "h1": h1, "h_1": h_1, "h2": h2, "h3": h3, "h_3": h_3, "h4": h4, "h_4": h_4, "h5": h5, "h6": h6, "h_6": h_6}
    fixed_parameters = constants["fixed_parameters"]
    loss = 0.0
    loss += jnp.mean(jnp.square((solution[:, 2] - dataset[:, 0]) / (jnp.max(dataset[:, 0]) - jnp.min(dataset[:, 0]) + 1e-12)))
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
    k1, k_1, k2, k3, k_3, k4, h1, h_1, h2, h3, h_3, h4, h_4, h5, h6, h_6 = unscaled_parameters
    trainable_parameters = {"k1": k1, "k_1": k_1, "k2": k2, "k3": k3, "k_3": k_3, "k4": k4, "h1": h1, "h_1": h_1, "h2": h2, "h3": h3, "h_3": h_3, "h4": h4, "h_4": h_4, "h5": h5, "h6": h6, "h_6": h_6}
    fixed_parameters = constants["fixed_parameters"]
    Nts = solution_time.shape[0]
    out = jnp.zeros((Nts, 13))
    out = out.at[:, 0].set(solution_time)
    out = out.at[:, 1].set(dataset[:, 0])
    out = out.at[:, 2].set(solution[:, 0])
    out = out.at[:, 3].set(solution[:, 1])
    out = out.at[:, 4].set(solution[:, 2])
    out = out.at[:, 5].set(solution[:, 3])
    out = out.at[:, 6].set(solution[:, 4])
    out = out.at[:, 7].set(solution[:, 5])
    out = out.at[:, 8].set(solution[:, 6])
    out = out.at[:, 9].set(solution[:, 7])
    out = out.at[:, 10].set(solution[:, 8])
    out = out.at[:, 11].set(solution[:, 9])
    out = out.at[:, 12].set(solution[:, 10])
    return out

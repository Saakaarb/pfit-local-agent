# pfit-sources: pending=true
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
    EGFR_turnover = trainable_parameters['EGFR_turnover']
    reaction_1_k1 = trainable_parameters['reaction_1_k1']
    reaction_1_k2 = trainable_parameters['reaction_1_k2']
    reaction_2_k1 = trainable_parameters['reaction_2_k1']
    reaction_2_k2 = trainable_parameters['reaction_2_k2']
    reaction_3_k1 = trainable_parameters['reaction_3_k1']
    reaction_4_k1 = trainable_parameters['reaction_4_k1']
    reaction_5_k1 = trainable_parameters['reaction_5_k1']
    reaction_5_k2 = trainable_parameters['reaction_5_k2']
    reaction_6_k1 = trainable_parameters['reaction_6_k1']
    reaction_7_k1 = trainable_parameters['reaction_7_k1']
    reaction_8_k1 = trainable_parameters['reaction_8_k1']
    reaction_9_k1 = trainable_parameters['reaction_9_k1']
    scaling_pEGFR_tot = trainable_parameters['scaling_pEGFR_tot']
    scaling_pAkt_tot = trainable_parameters['scaling_pAkt_tot']
    scaling_pS6_tot = trainable_parameters['scaling_pS6_tot']
    EGFR = solution[:, 0]
    EGF_EGFR = solution[:, 1]
    pEGFR = solution[:, 2]
    pEGFR_Akt = solution[:, 3]
    Akt = solution[:, 4]
    pAkt = solution[:, 5]
    pAkt_S6 = solution[:, 6]
    S6 = solution[:, 7]
    pS6 = solution[:, 8]
    L = solution[:, 9]
    return {'pEGFR_tot': scaling_pEGFR_tot * (pEGFR + pEGFR_Akt), 'pAkt_tot': scaling_pAkt_tot * (pAkt + pAkt_S6), 'pS6_tot': scaling_pS6_tot * pS6}

@jax.jit
def user_defined_system(t, y, other_args):
    constants = other_args["constants"]
    trainable_variables = other_args["trainable_variables"]
    dataset = constants["dataset"]
    t_eval = constants["t_eval"]
    unscaled_parameters = unscale_value(trainable_variables, constants["min_limits"], constants["max_limits"], constants["is_logscale"])
    EGFR_turnover, reaction_1_k1, reaction_1_k2, reaction_2_k1, reaction_2_k2, reaction_3_k1, reaction_4_k1, reaction_5_k1, reaction_5_k2, reaction_6_k1, reaction_7_k1, reaction_8_k1, reaction_9_k1, scaling_pEGFR_tot, scaling_pAkt_tot, scaling_pS6_tot = unscaled_parameters
    trainable_parameters = {"EGFR_turnover": EGFR_turnover, "reaction_1_k1": reaction_1_k1, "reaction_1_k2": reaction_1_k2, "reaction_2_k1": reaction_2_k1, "reaction_2_k2": reaction_2_k2, "reaction_3_k1": reaction_3_k1, "reaction_4_k1": reaction_4_k1, "reaction_5_k1": reaction_5_k1, "reaction_5_k2": reaction_5_k2, "reaction_6_k1": reaction_6_k1, "reaction_7_k1": reaction_7_k1, "reaction_8_k1": reaction_8_k1, "reaction_9_k1": reaction_9_k1, "scaling_pEGFR_tot": scaling_pEGFR_tot, "scaling_pAkt_tot": scaling_pAkt_tot, "scaling_pS6_tot": scaling_pS6_tot}
    fixed_parameters = constants["fixed_parameters"]
    EGFR = y[0]
    EGF_EGFR = y[1]
    pEGFR = y[2]
    pEGFR_Akt = y[3]
    Akt = y[4]
    pAkt = y[5]
    pAkt_S6 = y[6]
    S6 = y[7]
    pS6 = y[8]
    L = y[9]

    dEGFRdt = -(reaction_1_k1 * L * EGFR - reaction_1_k2 * EGF_EGFR) - EGFR_turnover * EGFR + 68190.0 * EGFR_turnover
    dEGF_EGFRdt = reaction_1_k1 * L * EGFR - reaction_1_k2 * EGF_EGFR - reaction_9_k1 * EGF_EGFR
    dpEGFRdt = -(reaction_2_k1 * Akt * pEGFR - reaction_2_k2 * pEGFR_Akt) + reaction_3_k1 * pEGFR_Akt - reaction_4_k1 * pEGFR + reaction_9_k1 * EGF_EGFR
    dpEGFR_Aktdt = reaction_2_k1 * Akt * pEGFR - reaction_2_k2 * pEGFR_Akt - reaction_3_k1 * pEGFR_Akt
    dAktdt = -(reaction_2_k1 * Akt * pEGFR - reaction_2_k2 * pEGFR_Akt) + reaction_7_k1 * pAkt
    dpAktdt = reaction_3_k1 * pEGFR_Akt - (reaction_5_k1 * S6 * pAkt - reaction_5_k2 * pAkt_S6) + reaction_6_k1 * pAkt_S6 - reaction_7_k1 * pAkt
    dpAkt_S6dt = reaction_5_k1 * S6 * pAkt - reaction_5_k2 * pAkt_S6 - reaction_6_k1 * pAkt_S6
    dS6dt = -(reaction_5_k1 * S6 * pAkt - reaction_5_k2 * pAkt_S6) + reaction_8_k1 * pS6
    dpS6dt = reaction_6_k1 * pAkt_S6 - reaction_8_k1 * pS6
    dLdt = 0.0
    return jnp.array([dEGFRdt, dEGF_EGFRdt, dpEGFRdt, dpEGFR_Aktdt, dAktdt, dpAktdt, dpAkt_S6dt, dS6dt, dpS6dt, dLdt])

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
        max_steps=20000,
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
    EGFR_turnover, reaction_1_k1, reaction_1_k2, reaction_2_k1, reaction_2_k2, reaction_3_k1, reaction_4_k1, reaction_5_k1, reaction_5_k2, reaction_6_k1, reaction_7_k1, reaction_8_k1, reaction_9_k1, scaling_pEGFR_tot, scaling_pAkt_tot, scaling_pS6_tot = unscaled_parameters
    trainable_parameters = {"EGFR_turnover": EGFR_turnover, "reaction_1_k1": reaction_1_k1, "reaction_1_k2": reaction_1_k2, "reaction_2_k1": reaction_2_k1, "reaction_2_k2": reaction_2_k2, "reaction_3_k1": reaction_3_k1, "reaction_4_k1": reaction_4_k1, "reaction_5_k1": reaction_5_k1, "reaction_5_k2": reaction_5_k2, "reaction_6_k1": reaction_6_k1, "reaction_7_k1": reaction_7_k1, "reaction_8_k1": reaction_8_k1, "reaction_9_k1": reaction_9_k1, "scaling_pEGFR_tot": scaling_pEGFR_tot, "scaling_pAkt_tot": scaling_pAkt_tot, "scaling_pS6_tot": scaling_pS6_tot}
    fixed_parameters = constants["fixed_parameters"]
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    loss = 0.0
    loss += jnp.mean(jnp.square((observables['pEGFR_tot'] - dataset[:, 0]) / (jnp.max(jnp.abs(dataset[:, 0])) + 1e-12)))
    loss += jnp.mean(jnp.square((observables['pAkt_tot'] - dataset[:, 1]) / (jnp.max(jnp.abs(dataset[:, 1])) + 1e-12)))
    loss += jnp.mean(jnp.square((observables['pS6_tot'] - dataset[:, 2]) / (jnp.max(jnp.abs(dataset[:, 2])) + 1e-12)))
    loss = jnp.sqrt(loss / 3)
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
    EGFR_turnover, reaction_1_k1, reaction_1_k2, reaction_2_k1, reaction_2_k2, reaction_3_k1, reaction_4_k1, reaction_5_k1, reaction_5_k2, reaction_6_k1, reaction_7_k1, reaction_8_k1, reaction_9_k1, scaling_pEGFR_tot, scaling_pAkt_tot, scaling_pS6_tot = unscaled_parameters
    trainable_parameters = {"EGFR_turnover": EGFR_turnover, "reaction_1_k1": reaction_1_k1, "reaction_1_k2": reaction_1_k2, "reaction_2_k1": reaction_2_k1, "reaction_2_k2": reaction_2_k2, "reaction_3_k1": reaction_3_k1, "reaction_4_k1": reaction_4_k1, "reaction_5_k1": reaction_5_k1, "reaction_5_k2": reaction_5_k2, "reaction_6_k1": reaction_6_k1, "reaction_7_k1": reaction_7_k1, "reaction_8_k1": reaction_8_k1, "reaction_9_k1": reaction_9_k1, "scaling_pEGFR_tot": scaling_pEGFR_tot, "scaling_pAkt_tot": scaling_pAkt_tot, "scaling_pS6_tot": scaling_pS6_tot}
    fixed_parameters = constants["fixed_parameters"]
    writeout_array = np.zeros([solution_time.shape[0], 17])
    writeout_array[:, 0] = solution_time
    writeout_array[:, 1] = dataset[:, 0]
    writeout_array[:, 2] = dataset[:, 1]
    writeout_array[:, 3] = dataset[:, 2]
    writeout_array[:, 4] = solution[:, 0]
    writeout_array[:, 5] = solution[:, 1]
    writeout_array[:, 6] = solution[:, 2]
    writeout_array[:, 7] = solution[:, 3]
    writeout_array[:, 8] = solution[:, 4]
    writeout_array[:, 9] = solution[:, 5]
    writeout_array[:, 10] = solution[:, 6]
    writeout_array[:, 11] = solution[:, 7]
    writeout_array[:, 12] = solution[:, 8]
    writeout_array[:, 13] = solution[:, 9]
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    writeout_array[:, 14] = observables['pEGFR_tot']
    writeout_array[:, 15] = observables['pAkt_tot']
    writeout_array[:, 16] = observables['pS6_tot']
    return writeout_array

# pfit-sources: user_model.py=1ccd0faf8c454423 user_input.yaml=090793136577b560
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
    ka1 = trainable_parameters['ka1']
    ka2fold = trainable_parameters['ka2fold']
    kd1 = trainable_parameters['kd1']
    kd2fold = trainable_parameters['kd2fold']
    kin = trainable_parameters['kin']
    kin2 = trainable_parameters['kin2']
    kout = trainable_parameters['kout']
    kout2 = trainable_parameters['kout2']
    kout_frag = trainable_parameters['kout_frag']
    kon_unspec = trainable_parameters['kon_unspec']
    koff_unspec = trainable_parameters['koff_unspec']
    offset = trainable_parameters['offset']
    km = trainable_parameters['km']
    scale = trainable_parameters['scale']
    fragments = trainable_parameters['fragments']
    Ins = solution[:, 0]
    Rec1 = solution[:, 1]
    Rec2 = solution[:, 2]
    IR1 = solution[:, 3]
    IR2 = solution[:, 4]
    IR1in = solution[:, 5]
    IR2in = solution[:, 6]
    Uptake1 = solution[:, 7]
    Uptake2 = solution[:, 8]
    InsulinFragments = solution[:, 9]
    BoundUnspec = solution[:, 10]
    return {'Insulin_signal': offset + scale * (Ins + fragments * InsulinFragments) / (1 + (Ins + fragments * InsulinFragments) / km)}

@jax.jit
def user_defined_system(t, y, other_args):
    constants = other_args["constants"]
    trainable_variables = other_args["trainable_variables"]
    dataset = constants["dataset"]
    t_eval = constants["t_eval"]
    unscaled_parameters = unscale_value(trainable_variables, constants["min_limits"], constants["max_limits"], constants["is_logscale"])
    ka1, ka2fold, kd1, kd2fold, kin, kin2, kout, kout2, kout_frag, kon_unspec, koff_unspec, offset, km, scale, fragments = unscaled_parameters
    trainable_parameters = {"ka1": ka1, "ka2fold": ka2fold, "kd1": kd1, "kd2fold": kd2fold, "kin": kin, "kin2": kin2, "kout": kout, "kout2": kout2, "kout_frag": kout_frag, "kon_unspec": kon_unspec, "koff_unspec": koff_unspec, "offset": offset, "km": km, "scale": scale, "fragments": fragments}
    fixed_parameters = constants["fixed_parameters"]
    Ins = y[0]
    Rec1 = y[1]
    Rec2 = y[2]
    IR1 = y[3]
    IR2 = y[4]
    IR1in = y[5]
    IR2in = y[6]
    Uptake1 = y[7]
    Uptake2 = y[8]
    InsulinFragments = y[9]
    BoundUnspec = y[10]

    dInsdt = -(ka1 * Ins * Rec1) - ka1 * ka2fold * Ins * Rec2 - kon_unspec * Ins + koff_unspec * BoundUnspec + kd1 * IR1 + kd1 * kd2fold * IR2
    dRec1dt = -(ka1 * Ins * Rec1) + kd1 * IR1 + kout_frag * IR1in
    dRec2dt = -(ka1 * ka2fold * Ins * Rec2) + kd1 * kd2fold * IR2 + kout_frag * IR2in
    dIR1dt = ka1 * Ins * Rec1 - kd1 * IR1 - kin * IR1 + kout * IR1in
    dIR2dt = ka1 * ka2fold * Ins * Rec2 - kd1 * kd2fold * IR2 - kin2 * IR2 + kout2 * IR2in
    dIR1indt = kin * IR1 - kout * IR1in - kout_frag * IR1in
    dIR2indt = kin2 * IR2 - kout2 * IR2in - kout_frag * IR2in
    dUptake1dt = ka1 * Ins * Rec1 - kd1 * IR1
    dUptake2dt = ka1 * ka2fold * Ins * Rec2 - kd1 * kd2fold * IR2
    dInsulinFragmentsdt = kout_frag * IR1in + kout_frag * IR2in
    dBoundUnspecdt = kon_unspec * Ins - koff_unspec * BoundUnspec
    return jnp.array([dInsdt, dRec1dt, dRec2dt, dIR1dt, dIR2dt, dIR1indt, dIR2indt, dUptake1dt, dUptake2dt, dInsulinFragmentsdt, dBoundUnspecdt])

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
    ka1, ka2fold, kd1, kd2fold, kin, kin2, kout, kout2, kout_frag, kon_unspec, koff_unspec, offset, km, scale, fragments = unscaled_parameters
    trainable_parameters = {"ka1": ka1, "ka2fold": ka2fold, "kd1": kd1, "kd2fold": kd2fold, "kin": kin, "kin2": kin2, "kout": kout, "kout2": kout2, "kout_frag": kout_frag, "kon_unspec": kon_unspec, "koff_unspec": koff_unspec, "offset": offset, "km": km, "scale": scale, "fragments": fragments}
    fixed_parameters = constants["fixed_parameters"]
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    loss = 0.0
    eps_Insulin_signal = jnp.min(jnp.where(dataset[:, 0] > 0.0, dataset[:, 0], jnp.inf))
    log_sim_Insulin_signal = jnp.log10(observables['Insulin_signal'] + eps_Insulin_signal)
    log_measured_Insulin_signal = jnp.log10(dataset[:, 0] + eps_Insulin_signal)
    scale_log_Insulin_signal = jnp.max(log_measured_Insulin_signal) - jnp.min(log_measured_Insulin_signal) + 1e-12
    loss += jnp.mean(jnp.square((log_sim_Insulin_signal - log_measured_Insulin_signal) / scale_log_Insulin_signal))
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
    ka1, ka2fold, kd1, kd2fold, kin, kin2, kout, kout2, kout_frag, kon_unspec, koff_unspec, offset, km, scale, fragments = unscaled_parameters
    trainable_parameters = {"ka1": ka1, "ka2fold": ka2fold, "kd1": kd1, "kd2fold": kd2fold, "kin": kin, "kin2": kin2, "kout": kout, "kout2": kout2, "kout_frag": kout_frag, "kon_unspec": kon_unspec, "koff_unspec": koff_unspec, "offset": offset, "km": km, "scale": scale, "fragments": fragments}
    fixed_parameters = constants["fixed_parameters"]
    writeout_array = np.zeros([solution_time.shape[0], 14])
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
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    writeout_array[:, 13] = observables['Insulin_signal']
    return writeout_array

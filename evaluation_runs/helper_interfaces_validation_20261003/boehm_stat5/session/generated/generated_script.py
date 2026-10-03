# pfit-sources: user_model.py=af47f5db7c3a6055 user_input.yaml=286350229bb5e877
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
    k_deg = trainable_parameters['k_deg']
    k_exp_hetero = trainable_parameters['k_exp_hetero']
    k_exp_homo = trainable_parameters['k_exp_homo']
    k_imp_hetero = trainable_parameters['k_imp_hetero']
    k_imp_homo = trainable_parameters['k_imp_homo']
    k_phos = trainable_parameters['k_phos']
    specC17 = fixed_parameters['specC17']
    Epo0 = fixed_parameters['Epo0']
    cyt = fixed_parameters['cyt']
    nuc = fixed_parameters['nuc']
    A = solution[:, 0]
    B = solution[:, 1]
    ApB = solution[:, 2]
    ApA = solution[:, 3]
    BpB = solution[:, 4]
    nApA = solution[:, 5]
    nApB = solution[:, 6]
    nBpB = solution[:, 7]
    return {'pSTAT5A': (100 * ApB + 200 * ApA * specC17) / (ApB + A * specC17 + 2 * ApA * specC17), 'pSTAT5B': -(100 * ApB - 200 * BpB * (specC17 - 1)) / (B * (specC17 - 1) - ApB + 2 * BpB * (specC17 - 1)), 'rSTAT5A': (100 * ApB + 100 * A * specC17 + 200 * ApA * specC17) / (2 * ApB + A * specC17 + 2 * ApA * specC17 - B * (specC17 - 1) - 2 * BpB * (specC17 - 1))}

@jax.jit
def user_defined_system(t, y, other_args):
    constants = other_args["constants"]
    trainable_variables = other_args["trainable_variables"]
    dataset = constants["dataset"]
    t_eval = constants["t_eval"]
    unscaled_parameters = unscale_value(trainable_variables, constants["min_limits"], constants["max_limits"], constants["is_logscale"])
    k_deg, k_exp_hetero, k_exp_homo, k_imp_hetero, k_imp_homo, k_phos = unscaled_parameters
    trainable_parameters = {"k_deg": k_deg, "k_exp_hetero": k_exp_hetero, "k_exp_homo": k_exp_homo, "k_imp_hetero": k_imp_hetero, "k_imp_homo": k_imp_homo, "k_phos": k_phos}
    fixed_parameters = constants["fixed_parameters"]
    specC17 = fixed_parameters['specC17']
    Epo0 = fixed_parameters['Epo0']
    cyt = fixed_parameters['cyt']
    nuc = fixed_parameters['nuc']
    A = y[0]
    B = y[1]
    ApB = y[2]
    ApA = y[3]
    BpB = y[4]
    nApA = y[5]
    nApB = y[6]
    nBpB = y[7]

    dAdt = -2 * (k_phos * (Epo0 * jnp.exp(-k_deg * t)) * A * A) - k_phos * (Epo0 * jnp.exp(-k_deg * t)) * A * B + nuc / cyt * (2 * k_exp_homo * nApA + k_exp_hetero * nApB)
    dBdt = -2 * (k_phos * (Epo0 * jnp.exp(-k_deg * t)) * B * B) - k_phos * (Epo0 * jnp.exp(-k_deg * t)) * A * B + nuc / cyt * (2 * k_exp_homo * nBpB + k_exp_hetero * nApB)
    dApBdt = k_phos * (Epo0 * jnp.exp(-k_deg * t)) * A * B - k_imp_hetero * ApB
    dApAdt = k_phos * (Epo0 * jnp.exp(-k_deg * t)) * A * A - k_imp_homo * ApA
    dBpBdt = k_phos * (Epo0 * jnp.exp(-k_deg * t)) * B * B - k_imp_homo * BpB
    dnApAdt = cyt / nuc * k_imp_homo * ApA - k_exp_homo * nApA
    dnApBdt = cyt / nuc * k_imp_hetero * ApB - k_exp_hetero * nApB
    dnBpBdt = cyt / nuc * k_imp_homo * BpB - k_exp_homo * nBpB
    return jnp.array([dAdt, dBdt, dApBdt, dApAdt, dBpBdt, dnApAdt, dnApBdt, dnBpBdt])

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
    k_deg, k_exp_hetero, k_exp_homo, k_imp_hetero, k_imp_homo, k_phos = unscaled_parameters
    trainable_parameters = {"k_deg": k_deg, "k_exp_hetero": k_exp_hetero, "k_exp_homo": k_exp_homo, "k_imp_hetero": k_imp_hetero, "k_imp_homo": k_imp_homo, "k_phos": k_phos}
    fixed_parameters = constants["fixed_parameters"]
    specC17 = fixed_parameters['specC17']
    Epo0 = fixed_parameters['Epo0']
    cyt = fixed_parameters['cyt']
    nuc = fixed_parameters['nuc']
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    loss = 0.0
    loss += jnp.mean(jnp.square((observables['pSTAT5A'] - dataset[:, 0]) / (jnp.max(jnp.abs(dataset[:, 0])) + 1e-12)))
    loss += jnp.mean(jnp.square((observables['pSTAT5B'] - dataset[:, 1]) / (jnp.max(jnp.abs(dataset[:, 1])) + 1e-12)))
    loss += jnp.mean(jnp.square((observables['rSTAT5A'] - dataset[:, 2]) / (jnp.max(jnp.abs(dataset[:, 2])) + 1e-12)))
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
    k_deg, k_exp_hetero, k_exp_homo, k_imp_hetero, k_imp_homo, k_phos = unscaled_parameters
    trainable_parameters = {"k_deg": k_deg, "k_exp_hetero": k_exp_hetero, "k_exp_homo": k_exp_homo, "k_imp_hetero": k_imp_hetero, "k_imp_homo": k_imp_homo, "k_phos": k_phos}
    fixed_parameters = constants["fixed_parameters"]
    specC17 = fixed_parameters['specC17']
    Epo0 = fixed_parameters['Epo0']
    cyt = fixed_parameters['cyt']
    nuc = fixed_parameters['nuc']
    writeout_array = np.zeros([solution_time.shape[0], 15])
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
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    writeout_array[:, 12] = observables['pSTAT5A']
    writeout_array[:, 13] = observables['pSTAT5B']
    writeout_array[:, 14] = observables['rSTAT5A']
    return writeout_array

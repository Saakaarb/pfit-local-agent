# pfit-sources: user_model.py=270f188c3b1822c3 user_input.yaml=f4893115bb9229d1
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
    Kon_IL13Rec, Rec_intern, Rec_recycle, Rec_phosphorylation, JAK2_phosphorylation, JAK2_p_inhibition, pRec_intern, pRec_degradation, pJAK2_dephosphorylation, STAT5_phosphorylation, pSTAT5_dephosphorylation, SOCS3mRNA_production, SOCS3_translation, SOCS3_accumulation, SOCS3_degradation = unscaled_parameters
    trainable_parameters = {"Kon_IL13Rec": Kon_IL13Rec, "Rec_intern": Rec_intern, "Rec_recycle": Rec_recycle, "Rec_phosphorylation": Rec_phosphorylation, "JAK2_phosphorylation": JAK2_phosphorylation, "JAK2_p_inhibition": JAK2_p_inhibition, "pRec_intern": pRec_intern, "pRec_degradation": pRec_degradation, "pJAK2_dephosphorylation": pJAK2_dephosphorylation, "STAT5_phosphorylation": STAT5_phosphorylation, "pSTAT5_dephosphorylation": pSTAT5_dephosphorylation, "SOCS3mRNA_production": SOCS3mRNA_production, "SOCS3_translation": SOCS3_translation, "SOCS3_accumulation": SOCS3_accumulation, "SOCS3_degradation": SOCS3_degradation}
    fixed_parameters = constants["fixed_parameters"]
    DecoyR_binding = fixed_parameters['DecoyR_binding']
    CD274mRNA_production = fixed_parameters['CD274mRNA_production']
    Rec = y[0]
    Rec_i = y[1]
    IL13_Rec = y[2]
    p_IL13_Rec = y[3]
    p_IL13_Rec_i = y[4]
    JAK2 = y[5]
    pJAK2 = y[6]
    STAT5 = y[7]
    pSTAT5 = y[8]
    SOCS3mRNA = y[9]
    DecoyR = y[10]
    IL13_DecoyR = y[11]
    SOCS3 = y[12]
    CD274mRNA = y[13]
    il13_level = y[14]

    dRecdt = -(2.265 * il13_level * Kon_IL13Rec * Rec) - Rec * Rec_intern + Rec_i * Rec_recycle
    dRec_idt = Rec * Rec_intern - Rec_i * Rec_recycle
    dIL13_Recdt = 2.265 * il13_level * Kon_IL13Rec * Rec - IL13_Rec * Rec_phosphorylation * pJAK2
    dp_IL13_Recdt = IL13_Rec * Rec_phosphorylation * pJAK2 - pRec_intern * p_IL13_Rec
    dp_IL13_Rec_idt = pRec_intern * p_IL13_Rec - pRec_degradation * p_IL13_Rec_i
    dJAK2dt = -(IL13_Rec * JAK2 * JAK2_phosphorylation / (1.0 + JAK2_p_inhibition * SOCS3)) - JAK2 * JAK2_phosphorylation * p_IL13_Rec / (1.0 + JAK2_p_inhibition * SOCS3) + 91.0 * pJAK2 * pJAK2_dephosphorylation
    dpJAK2dt = IL13_Rec * JAK2 * JAK2_phosphorylation / (1.0 + JAK2_p_inhibition * SOCS3) + JAK2 * JAK2_phosphorylation * p_IL13_Rec / (1.0 + JAK2_p_inhibition * SOCS3) - 91.0 * pJAK2 * pJAK2_dephosphorylation
    dSTAT5dt = -(STAT5 * STAT5_phosphorylation * pJAK2) + 91.0 * pSTAT5 * pSTAT5_dephosphorylation
    dpSTAT5dt = STAT5 * STAT5_phosphorylation * pJAK2 - 91.0 * pSTAT5 * pSTAT5_dephosphorylation
    dSOCS3mRNAdt = SOCS3mRNA_production * pSTAT5
    dDecoyRdt = -(2.265 * il13_level * DecoyR * DecoyR_binding)
    dIL13_DecoyRdt = 2.265 * il13_level * DecoyR * DecoyR_binding
    dSOCS3dt = SOCS3mRNA * SOCS3_translation / (SOCS3mRNA + SOCS3_accumulation) - SOCS3 * SOCS3_degradation
    dCD274mRNAdt = CD274mRNA_production * pSTAT5
    dil13_leveldt = 0.0
    return jnp.array([dRecdt, dRec_idt, dIL13_Recdt, dp_IL13_Recdt, dp_IL13_Rec_idt, dJAK2dt, dpJAK2dt, dSTAT5dt, dpSTAT5dt, dSOCS3mRNAdt, dDecoyRdt, dIL13_DecoyRdt, dSOCS3dt, dCD274mRNAdt, dil13_leveldt])

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
        max_steps=4000,
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
    Kon_IL13Rec, Rec_intern, Rec_recycle, Rec_phosphorylation, JAK2_phosphorylation, JAK2_p_inhibition, pRec_intern, pRec_degradation, pJAK2_dephosphorylation, STAT5_phosphorylation, pSTAT5_dephosphorylation, SOCS3mRNA_production, SOCS3_translation, SOCS3_accumulation, SOCS3_degradation = unscaled_parameters
    trainable_parameters = {"Kon_IL13Rec": Kon_IL13Rec, "Rec_intern": Rec_intern, "Rec_recycle": Rec_recycle, "Rec_phosphorylation": Rec_phosphorylation, "JAK2_phosphorylation": JAK2_phosphorylation, "JAK2_p_inhibition": JAK2_p_inhibition, "pRec_intern": pRec_intern, "pRec_degradation": pRec_degradation, "pJAK2_dephosphorylation": pJAK2_dephosphorylation, "STAT5_phosphorylation": STAT5_phosphorylation, "pSTAT5_dephosphorylation": pSTAT5_dephosphorylation, "SOCS3mRNA_production": SOCS3mRNA_production, "SOCS3_translation": SOCS3_translation, "SOCS3_accumulation": SOCS3_accumulation, "SOCS3_degradation": SOCS3_degradation}
    fixed_parameters = constants["fixed_parameters"]
    DecoyR_binding = fixed_parameters['DecoyR_binding']
    CD274mRNA_production = fixed_parameters['CD274mRNA_production']
    loss = 0.0
    loss += jnp.mean(jnp.square((solution[:, 8] - dataset[:, 0]) / (jnp.max(jnp.abs(dataset[:, 0])) + 1e-12)))
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
    Kon_IL13Rec, Rec_intern, Rec_recycle, Rec_phosphorylation, JAK2_phosphorylation, JAK2_p_inhibition, pRec_intern, pRec_degradation, pJAK2_dephosphorylation, STAT5_phosphorylation, pSTAT5_dephosphorylation, SOCS3mRNA_production, SOCS3_translation, SOCS3_accumulation, SOCS3_degradation = unscaled_parameters
    trainable_parameters = {"Kon_IL13Rec": Kon_IL13Rec, "Rec_intern": Rec_intern, "Rec_recycle": Rec_recycle, "Rec_phosphorylation": Rec_phosphorylation, "JAK2_phosphorylation": JAK2_phosphorylation, "JAK2_p_inhibition": JAK2_p_inhibition, "pRec_intern": pRec_intern, "pRec_degradation": pRec_degradation, "pJAK2_dephosphorylation": pJAK2_dephosphorylation, "STAT5_phosphorylation": STAT5_phosphorylation, "pSTAT5_dephosphorylation": pSTAT5_dephosphorylation, "SOCS3mRNA_production": SOCS3mRNA_production, "SOCS3_translation": SOCS3_translation, "SOCS3_accumulation": SOCS3_accumulation, "SOCS3_degradation": SOCS3_degradation}
    fixed_parameters = constants["fixed_parameters"]
    DecoyR_binding = fixed_parameters['DecoyR_binding']
    CD274mRNA_production = fixed_parameters['CD274mRNA_production']
    writeout_array = np.zeros([solution_time.shape[0], 17])
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
    writeout_array[:, 13] = solution[:, 11]
    writeout_array[:, 14] = solution[:, 12]
    writeout_array[:, 15] = solution[:, 13]
    writeout_array[:, 16] = solution[:, 14]
    return writeout_array

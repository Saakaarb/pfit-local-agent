import numpy as np

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
    return {
        'pEGFR_tot': scaling_pEGFR_tot * (pEGFR + pEGFR_Akt),
        'pAkt_tot': scaling_pAkt_tot * (pAkt + pAkt_S6),
        'pS6_tot': scaling_pS6_tot * pS6,
    }


def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
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

    dEGFRdt = -(reaction_1_k1 * L * EGFR - reaction_1_k2 * EGF_EGFR) - EGFR_turnover * EGFR + 68190 * EGFR_turnover
    dEGF_EGFRdt = reaction_1_k1 * L * EGFR - reaction_1_k2 * EGF_EGFR - reaction_9_k1 * EGF_EGFR
    dpEGFRdt = -(reaction_2_k1 * Akt * pEGFR - reaction_2_k2 * pEGFR_Akt) + reaction_3_k1 * pEGFR_Akt - reaction_4_k1 * pEGFR + reaction_9_k1 * EGF_EGFR
    dpEGFR_Aktdt = reaction_2_k1 * Akt * pEGFR - reaction_2_k2 * pEGFR_Akt - reaction_3_k1 * pEGFR_Akt
    dAktdt = -(reaction_2_k1 * Akt * pEGFR - reaction_2_k2 * pEGFR_Akt) + reaction_7_k1 * pAkt
    dpAktdt = reaction_3_k1 * pEGFR_Akt - (reaction_5_k1 * S6 * pAkt - reaction_5_k2 * pAkt_S6) + reaction_6_k1 * pAkt_S6 - reaction_7_k1 * pAkt
    dpAkt_S6dt = reaction_5_k1 * S6 * pAkt - reaction_5_k2 * pAkt_S6 - reaction_6_k1 * pAkt_S6
    dS6dt = -(reaction_5_k1 * S6 * pAkt - reaction_5_k2 * pAkt_S6) + reaction_8_k1 * pS6
    dpS6dt = reaction_6_k1 * pAkt_S6 - reaction_8_k1 * pS6
    dLdt = 0
    return np.array([dEGFRdt, dEGF_EGFRdt, dpEGFRdt, dpEGFR_Aktdt, dAktdt, dpAktdt, dpAkt_S6dt, dS6dt, dpS6dt, dLdt])

def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    loss = 0.0
    loss += np.mean(np.square((observables['pEGFR_tot'] - dataset[:, 0]) / (np.max(np.abs(dataset[:, 0])) + 1e-12)))
    loss += np.mean(np.square((observables['pAkt_tot'] - dataset[:, 1]) / (np.max(np.abs(dataset[:, 1])) + 1e-12)))
    loss += np.mean(np.square((observables['pS6_tot'] - dataset[:, 2]) / (np.max(np.abs(dataset[:, 2])) + 1e-12)))
    loss = np.sqrt(loss / 3)
    return float(loss)

def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
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

import numpy as np

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
    return {
        'pSTAT5A': (100 * ApB + 200 * ApA * specC17) / (ApB + A * specC17 + 2 * ApA * specC17),
        'pSTAT5B': -(100 * ApB - 200 * BpB * (specC17 - 1)) / ((B * (specC17 - 1) - ApB) + 2 * BpB * (specC17 - 1)),
        'rSTAT5A': (100 * ApB + 100 * A * specC17 + 200 * ApA * specC17) / (2 * ApB + A * specC17 + 2 * ApA * specC17 - B * (specC17 - 1) - 2 * BpB * (specC17 - 1)),
    }


def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
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
    A = y[0]
    B = y[1]
    ApB = y[2]
    ApA = y[3]
    BpB = y[4]
    nApA = y[5]
    nApB = y[6]
    nBpB = y[7]
    dAdt = -2 * (k_phos * (Epo0 * np.exp(-k_deg * t)) * A * A) - k_phos * (Epo0 * np.exp(-k_deg * t)) * A * B + nuc / cyt * (2 * k_exp_homo * nApA + k_exp_hetero * nApB)
    dBdt = -2 * (k_phos * (Epo0 * np.exp(-k_deg * t)) * B * B) - k_phos * (Epo0 * np.exp(-k_deg * t)) * A * B + nuc / cyt * (2 * k_exp_homo * nBpB + k_exp_hetero * nApB)
    dApBdt = k_phos * (Epo0 * np.exp(-k_deg * t)) * A * B - k_imp_hetero * ApB
    dApAdt = k_phos * (Epo0 * np.exp(-k_deg * t)) * A * A - k_imp_homo * ApA
    dBpBdt = k_phos * (Epo0 * np.exp(-k_deg * t)) * B * B - k_imp_homo * BpB
    dnApAdt = cyt / nuc * k_imp_homo * ApA - k_exp_homo * nApA
    dnApBdt = cyt / nuc * k_imp_hetero * ApB - k_exp_hetero * nApB
    dnBpBdt = cyt / nuc * k_imp_homo * BpB - k_exp_homo * nBpB
    return np.array([dAdt, dBdt, dApBdt, dApAdt, dBpBdt, dnApAdt, dnApBdt, dnBpBdt])

def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    loss = 0.0
    loss += np.mean(np.square((observables['pSTAT5A'] - dataset[:, 0]) / (np.max(dataset[:, 0]) - np.min(dataset[:, 0]) + 1e-12)))
    loss += np.mean(np.square((observables['pSTAT5B'] - dataset[:, 1]) / (np.max(dataset[:, 1]) - np.min(dataset[:, 1]) + 1e-12)))
    loss += np.mean(np.square((observables['rSTAT5A'] - dataset[:, 2]) / (np.max(dataset[:, 2]) - np.min(dataset[:, 2]) + 1e-12)))
    return float(loss)

def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
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

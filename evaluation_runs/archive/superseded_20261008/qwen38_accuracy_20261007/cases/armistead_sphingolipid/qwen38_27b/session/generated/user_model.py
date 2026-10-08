import numpy as np

def _observables(solution, trainable_parameters, fixed_parameters):
    k00 = trainable_parameters['k00']
    k0 = trainable_parameters['k0']
    k_d = trainable_parameters['k_d']
    k1 = trainable_parameters['k1']
    k2 = trainable_parameters['k2']
    k3 = trainable_parameters['k3']
    k4 = trainable_parameters['k4']
    k5 = trainable_parameters['k5']
    alpha_hai1a = trainable_parameters['alpha_hai1a']
    alpha_cer = trainable_parameters['alpha_cer']

    Sphinga = solution[:, 0]
    Cer = solution[:, 1]
    Sphingo = solution[:, 2]
    S1P = solution[:, 3]
    H = solution[:, 4]
    return {
        'Sphinga_obs': Sphinga,
        'Cer_obs': Cer,
        'Sphingo_obs': Sphingo,
    }


def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    k00 = trainable_parameters['k00']
    k0 = trainable_parameters['k0']
    k_d = trainable_parameters['k_d']
    k1 = trainable_parameters['k1']
    k2 = trainable_parameters['k2']
    k3 = trainable_parameters['k3']
    k4 = trainable_parameters['k4']
    k5 = trainable_parameters['k5']
    alpha_hai1a = trainable_parameters['alpha_hai1a']
    alpha_cer = trainable_parameters['alpha_cer']

    Sphinga = y[0]
    Cer = y[1]
    Sphingo = y[2]
    S1P = y[3]
    H = y[4]

    dSphingadt = k00 * (1 + H * alpha_cer) - k0 * Sphinga
    dCerdt = k0 * Sphinga - k1 * Cer + k2 * Sphingo - k_d * Cer
    dSphingodt = k1 * Cer - k2 * Sphingo - k3 * (1 - H * alpha_hai1a) * Sphingo + k4 * S1P
    dS1Pdt = k3 * (1 - H * alpha_hai1a) * Sphingo - k4 * S1P - k5 * S1P
    dHdt = 0
    return np.array([dSphingadt, dCerdt, dSphingodt, dS1Pdt, dHdt])

def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    loss = 0.0
    loss += np.mean(np.square((solution[:, 0] - dataset[:, 0]) / (np.max(np.abs(dataset[:, 0])) + 1e-12)))
    loss += np.mean(np.square((solution[:, 1] - dataset[:, 1]) / (np.max(np.abs(dataset[:, 1])) + 1e-12)))
    loss += np.mean(np.square((solution[:, 2] - dataset[:, 2]) / (np.max(np.abs(dataset[:, 2])) + 1e-12)))
    loss = np.sqrt(loss / 3)
    return float(loss)

def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    writeout_array = np.zeros([solution_time.shape[0], 12])
    writeout_array[:, 0] = solution_time
    writeout_array[:, 1] = dataset[:, 0]
    writeout_array[:, 2] = dataset[:, 1]
    writeout_array[:, 3] = dataset[:, 2]
    writeout_array[:, 4] = solution[:, 0]
    writeout_array[:, 5] = solution[:, 1]
    writeout_array[:, 6] = solution[:, 2]
    writeout_array[:, 7] = solution[:, 3]
    writeout_array[:, 8] = solution[:, 4]
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    writeout_array[:, 9] = observables['Sphinga_obs']
    writeout_array[:, 10] = observables['Cer_obs']
    writeout_array[:, 11] = observables['Sphingo_obs']
    return writeout_array

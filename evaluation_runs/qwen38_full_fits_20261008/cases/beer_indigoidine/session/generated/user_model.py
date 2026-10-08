import numpy as np

def _observables(solution, trainable_parameters, fixed_parameters):
    Bacmax = trainable_parameters['Bacmax']
    beta = trainable_parameters['beta']
    ksyn = trainable_parameters['ksyn']
    kdim = trainable_parameters['kdim']
    kdegi = trainable_parameters['kdegi']
    tau = fixed_parameters['tau']
    Bac = solution[:, 0]
    Glu = solution[:, 1]
    cGlu = solution[:, 2]
    Ind = solution[:, 3]
    return {
        'Bacnorm': Bac,
        'IndconcNormRange': Ind,
    }


def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    Bacmax = trainable_parameters['Bacmax']
    beta = trainable_parameters['beta']
    ksyn = trainable_parameters['ksyn']
    kdim = trainable_parameters['kdim']
    kdegi = trainable_parameters['kdegi']
    tau = fixed_parameters['tau']
    Bac = y[0]
    Glu = y[1]
    cGlu = y[2]
    Ind = y[3]

    dBacdt = beta * Bac * (0.5 * (1 + np.tanh((t - tau) / 0.01))) * (1 - Bac / Bacmax)
    dGludt = -ksyn * Bac * Glu
    dcGludt = ksyn * Bac * Glu - kdim * cGlu ** 2
    dInddt = kdim * cGlu ** 2 - kdegi * Ind
    return np.array([dBacdt, dGludt, dcGludt, dInddt])

def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    loss = 0.0
    loss += np.mean(np.square((solution[:, 0] - dataset[:, 0]) / (np.max(np.abs(dataset[:, 0])) + 1e-12)))
    loss += np.mean(np.square((solution[:, 3] - dataset[:, 1]) / (np.max(np.abs(dataset[:, 1])) + 1e-12)))
    loss = np.sqrt(loss / 2)
    return float(loss)

def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    writeout_array = np.zeros([solution_time.shape[0], 9])
    writeout_array[:, 0] = solution_time
    writeout_array[:, 1] = dataset[:, 0]
    writeout_array[:, 2] = dataset[:, 1]
    writeout_array[:, 3] = solution[:, 0]
    writeout_array[:, 4] = solution[:, 1]
    writeout_array[:, 5] = solution[:, 2]
    writeout_array[:, 6] = solution[:, 3]
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    writeout_array[:, 7] = observables['Bacnorm']
    writeout_array[:, 8] = observables['IndconcNormRange']
    return writeout_array

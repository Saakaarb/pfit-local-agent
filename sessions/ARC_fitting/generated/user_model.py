import numpy as np

def _observables(solution, trainable_parameters, fixed_parameters):
    Ea1 = trainable_parameters['Ea1']
    h1 = trainable_parameters['h1']
    A1 = trainable_parameters['A1']
    A2 = trainable_parameters['A2']
    Ea2 = trainable_parameters['Ea2']
    h2 = trainable_parameters['h2']
    m2 = trainable_parameters['m2']
    n2 = trainable_parameters['n2']
    kb = fixed_parameters['kb']
    c1 = solution[:, 0]
    c2 = solution[:, 1]
    T = solution[:, 2]
    return {
        'dTdt': np.abs(h1 * (-A1 * np.exp(-Ea1 / (kb * T)) * c1)) + np.abs(h2 * (A2 * np.exp(-Ea2 / (kb * T)) * c2 ** n2 * (1 - c2) ** m2)),
    }


def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    Ea1 = trainable_parameters['Ea1']
    h1 = trainable_parameters['h1']
    A1 = trainable_parameters['A1']
    A2 = trainable_parameters['A2']
    Ea2 = trainable_parameters['Ea2']
    h2 = trainable_parameters['h2']
    m2 = trainable_parameters['m2']
    n2 = trainable_parameters['n2']
    kb = fixed_parameters['kb']
    c1 = y[0]
    c2 = y[1]
    T = y[2]
    dc1dt = -A1 * np.exp(-Ea1 / (kb * T)) * c1
    dc2dt = A2 * np.exp(-Ea2 / (kb * T)) * c2 ** n2 * (1 - c2) ** m2
    dTdt = np.abs(h1 * (-A1 * np.exp(-Ea1 / (kb * T)) * c1)) + np.abs(h2 * (A2 * np.exp(-Ea2 / (kb * T)) * c2 ** n2 * (1 - c2) ** m2))
    return np.array([dc1dt, dc2dt, dTdt])

def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    loss = 0.0
    loss += np.mean(np.square((solution[:, 2] - dataset[:, 0]) / (np.max(dataset[:, 0]) - np.min(dataset[:, 0]) + 1e-12)))
    eps_dTdt = np.min(np.where(dataset[:, 1] > 0.0, dataset[:, 1], np.inf))
    log_sim_dTdt = np.log10(observables['dTdt'] + eps_dTdt)
    log_measured_dTdt = np.log10(dataset[:, 1] + eps_dTdt)
    scale_log_dTdt = np.max(log_measured_dTdt) - np.min(log_measured_dTdt) + 1e-12
    loss += np.mean(np.square((log_sim_dTdt - log_measured_dTdt) / scale_log_dTdt))
    loss += 10000.0 / (1.0 + np.exp(-np.clip(1000.0 * (solution[-1, 0] - 0.02), -60.0, 60.0)))
    loss += 10000.0 / (1.0 + np.exp(-np.clip(1000.0 * (0.98 - solution[-1, 1]), -60.0, 60.0)))
    loss += 10000.0 / (1.0 + np.exp(-np.clip(1000.0 * (np.sqrt(np.square(solution[-1, 2] - dataset[-1, 0]) + 1e-12) - 50.0), -60.0, 60.0)))
    return float(loss)

def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    writeout_array = np.zeros([solution_time.shape[0], 7])
    writeout_array[:, 0] = solution_time
    writeout_array[:, 1] = dataset[:, 0]
    writeout_array[:, 2] = dataset[:, 1]
    writeout_array[:, 3] = solution[:, 0]
    writeout_array[:, 4] = solution[:, 1]
    writeout_array[:, 5] = solution[:, 2]
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    writeout_array[:, 6] = observables['dTdt']
    return writeout_array

import numpy as np

def _observables(solution, trainable_parameters, fixed_parameters):
    k1 = trainable_parameters['k1']
    k2 = trainable_parameters['k2']
    k3 = trainable_parameters['k3']
    k4 = trainable_parameters['k4']

    x1 = solution[:, 0]
    x2 = solution[:, 1]
    return {
        'level_lower': x2,
    }


def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    k1 = trainable_parameters['k1']
    k2 = trainable_parameters['k2']
    k3 = trainable_parameters['k3']
    k4 = trainable_parameters['k4']

    x1 = y[0]
    x2 = y[1]
    pump_voltage = np.interp(t, t_eval, dataset[:, 1])
    dx1dt = -k1 * np.sqrt(np.maximum(x1, 0)) + k4 * pump_voltage
    dx2dt = k2 * np.sqrt(np.maximum(x1, 0)) - k3 * np.sqrt(np.maximum(x2, 0))
    return np.array([dx1dt, dx2dt])

def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    loss = 0.0
    loss += np.mean(np.square((solution[:, 1] - dataset[:, 0]) / (np.max(np.abs(dataset[:, 0])) + 1e-12)))
    loss = np.sqrt(loss / 1)
    return float(loss)

def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    writeout_array = np.zeros([solution_time.shape[0], 5])
    writeout_array[:, 0] = solution_time
    writeout_array[:, 1] = dataset[:, 0]
    writeout_array[:, 2] = solution[:, 0]
    writeout_array[:, 3] = solution[:, 1]
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    writeout_array[:, 4] = observables['level_lower']
    return writeout_array

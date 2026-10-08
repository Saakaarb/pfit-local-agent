import numpy as np

def _observables(solution, trainable_parameters, fixed_parameters):
    k1 = trainable_parameters['k1']
    k2 = trainable_parameters['k2']
    k3 = trainable_parameters['k3']
    k4 = trainable_parameters['k4']
    k_1 = trainable_parameters['k_1']
    k_2 = trainable_parameters['k_2']
    k_3 = trainable_parameters['k_3']
    k_4 = trainable_parameters['k_4']
    l2 = trainable_parameters['l2']
    l4 = trainable_parameters['l4']
    l6 = trainable_parameters['l6']
    l_2 = trainable_parameters['l_2']
    l_4 = trainable_parameters['l_4']
    l_6 = trainable_parameters['l_6']

    O = solution[:, 0]
    R = solution[:, 1]
    I1 = solution[:, 2]
    S = solution[:, 3]
    A = solution[:, 4]
    I2 = solution[:, 5]
    IP3 = solution[:, 6]
    Ca = solution[:, 7]
    return {
        'open_probability': (0.9 * A + 0.1 * O) ** 4,
    }


def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    k1 = trainable_parameters['k1']
    k2 = trainable_parameters['k2']
    k3 = trainable_parameters['k3']
    k4 = trainable_parameters['k4']
    k_1 = trainable_parameters['k_1']
    k_2 = trainable_parameters['k_2']
    k_3 = trainable_parameters['k_3']
    k_4 = trainable_parameters['k_4']
    l2 = trainable_parameters['l2']
    l4 = trainable_parameters['l4']
    l6 = trainable_parameters['l6']
    l_2 = trainable_parameters['l_2']
    l_4 = trainable_parameters['l_4']
    l_6 = trainable_parameters['l_6']

    O = y[0]
    R = y[1]
    I1 = y[2]
    S = y[3]
    A = y[4]
    I2 = y[5]
    IP3 = y[6]
    Ca = y[7]

    dOdt = -((k_2 + l_4 * Ca) / (1 + Ca / (k_4 * l6 / (k4 * l_6))) * O) + (k2 * (k_2 * l4 / (k2 * l_4)) + l4 * Ca) / (k_2 * l4 / (k2 * l_4) + Ca * (1 + k_2 * l4 / (k2 * l_4) / (k_1 * l2 / (k1 * l_2)))) * IP3 * R - (k4 * (k_4 * l6 / (k4 * l_6)) + l6) * Ca / (k_4 * l6 / (k4 * l_6) + Ca) * O + k_1 * l2 / (k1 * l_2) * (k_4 + l_6) / (k_1 * l2 / (k1 * l_2) + Ca) * A - k3 * (k_4 * l6 / (k4 * l_6)) / (k_4 * l6 / (k4 * l_6) + Ca) * O + k_3 * S
    dRdt = (k_2 + l_4 * Ca) / (1 + Ca / (k_4 * l6 / (k4 * l_6))) * O - (k2 * (k_2 * l4 / (k2 * l_4)) + l4 * Ca) / (k_2 * l4 / (k2 * l_4) + Ca * (1 + k_2 * l4 / (k2 * l_4) / (k_1 * l2 / (k1 * l_2)))) * IP3 * R - (k1 * (k_1 * l2 / (k1 * l_2)) + l2) * Ca / (k_1 * l2 / (k1 * l_2) + Ca * (1 + k_1 * l2 / (k1 * l_2) / (k_2 * l4 / (k2 * l_4)))) * R + (k_1 + l_2) * I1
    dI1dt = (k1 * (k_1 * l2 / (k1 * l_2)) + l2) * Ca / (k_1 * l2 / (k1 * l_2) + Ca * (1 + k_1 * l2 / (k1 * l_2) / (k_2 * l4 / (k2 * l_4)))) * R - (k_1 + l_2) * I1
    dSdt = k3 * (k_4 * l6 / (k4 * l_6)) / (k_4 * l6 / (k4 * l_6) + Ca) * O - k_3 * S
    dAdt = (k4 * (k_4 * l6 / (k4 * l_6)) + l6) * Ca / (k_4 * l6 / (k4 * l_6) + Ca) * O - k_1 * l2 / (k1 * l_2) * (k_4 + l_6) / (k_1 * l2 / (k1 * l_2) + Ca) * A - (k1 * (k_1 * l2 / (k1 * l_2)) + l2) * Ca / (k_1 * l2 / (k1 * l_2) + Ca) * A + (k_1 + l_2) * I2
    dI2dt = (k1 * (k_1 * l2 / (k1 * l_2)) + l2) * Ca / (k_1 * l2 / (k1 * l_2) + Ca) * A - (k_1 + l_2) * I2
    dIP3dt = 0
    dCadt = 0
    return np.array([dOdt, dRdt, dI1dt, dSdt, dAdt, dI2dt, dIP3dt, dCadt])

def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    residuals = np.column_stack((
        observables['open_probability'] - dataset[:, 0],
    ))
    return float(np.mean(np.square(residuals)))

def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    writeout_array = np.zeros([solution_time.shape[0], 11])
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
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    writeout_array[:, 10] = observables['open_probability']
    return writeout_array

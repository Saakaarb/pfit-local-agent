import numpy as np

def _observables(solution, trainable_parameters, fixed_parameters):
    c2 = trainable_parameters['c2']
    Dk = trainable_parameters['Dk']
    Dc = trainable_parameters['Dc']
    m1 = trainable_parameters['m1']
    m2 = trainable_parameters['m2']
    vf = fixed_parameters['vf']
    x1 = solution[:, 0]
    x2 = solution[:, 1]
    v1 = solution[:, 2]
    v2 = solution[:, 3]
    k = solution[:, 4]
    c1 = solution[:, 5]
    return {
        'contact_force': np.abs(k * (x2 - x1) - c1 * np.abs(v1) * np.sign(v1)) / 1000,
        'displacement': x1,
    }


def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    c2 = trainable_parameters['c2']
    Dk = trainable_parameters['Dk']
    Dc = trainable_parameters['Dc']
    m1 = trainable_parameters['m1']
    m2 = trainable_parameters['m2']
    vf = fixed_parameters['vf']
    x1 = y[0]
    x2 = y[1]
    v1 = y[2]
    v2 = y[3]
    k = y[4]
    c1 = y[5]
    dx1dt = v1
    dx2dt = v2
    dv1dt = (k * (x2 - x1) - c1 * np.abs(v1) * np.sign(v1)) / m1
    dv2dt = 0.5 * (1 + np.tanh((x2 - x1) / 0.01)) * (-k * (x2 - x1) - c2 * np.sign(v2)) / m2
    dkdt = Dk * np.abs(m1 * v1 * (k * (x2 - x1) - c1 * np.abs(v1) * np.sign(v1)) / m1)
    dc1dt = Dc * np.abs(m1 * v1 * (k * (x2 - x1) - c1 * np.abs(v1) * np.sign(v1)) / m1)
    return np.array([dx1dt, dx2dt, dv1dt, dv2dt, dkdt, dc1dt])

def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    loss = 0.0
    loss += np.mean(np.square((observables['contact_force'] - dataset[:, 0]) / (np.max(dataset[:, 0]) - np.min(dataset[:, 0]) + 1e-12)))
    loss += np.mean(np.square((observables['displacement'] - dataset[:, 1]) / (np.max(dataset[:, 1]) - np.min(dataset[:, 1]) + 1e-12)))
    return float(loss)

def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    writeout_array = np.zeros([solution_time.shape[0], 11])
    writeout_array[:, 0] = solution_time
    writeout_array[:, 1] = dataset[:, 0]
    writeout_array[:, 2] = dataset[:, 1]
    writeout_array[:, 3] = solution[:, 0]
    writeout_array[:, 4] = solution[:, 1]
    writeout_array[:, 5] = solution[:, 2]
    writeout_array[:, 6] = solution[:, 3]
    writeout_array[:, 7] = solution[:, 4]
    writeout_array[:, 8] = solution[:, 5]
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    writeout_array[:, 9] = observables['contact_force']
    writeout_array[:, 10] = observables['displacement']
    return writeout_array

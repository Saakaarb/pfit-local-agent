import numpy as np

def _observables(solution, trainable_parameters, fixed_parameters):
    K2 = trainable_parameters['K2']
    K_par = trainable_parameters['K_par']
    Ka = trainable_parameters['Ka']
    Kd = trainable_parameters['Kd']
    Kf = trainable_parameters['Kf']
    Kp = trainable_parameters['Kp']
    Ky = trainable_parameters['Ky']
    Kz = trainable_parameters['Kz']
    Vd = trainable_parameters['Vd']
    Vm2 = trainable_parameters['Vm2']
    Vm3 = trainable_parameters['Vm3']
    Vp = trainable_parameters['Vp']
    beta_par = trainable_parameters['beta_par']
    epsilon_par = trainable_parameters['epsilon_par']
    n_par = trainable_parameters['n_par']
    v0 = trainable_parameters['v0']
    v1 = trainable_parameters['v1']
    offset = trainable_parameters['offset']
    scale = trainable_parameters['scale']

    Z = solution[:, 0]
    Y = solution[:, 1]
    A = solution[:, 2]
    return {
        'Ca': offset + scale * Z,
    }


def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    K2 = trainable_parameters['K2']
    K_par = trainable_parameters['K_par']
    Ka = trainable_parameters['Ka']
    Kd = trainable_parameters['Kd']
    Kf = trainable_parameters['Kf']
    Kp = trainable_parameters['Kp']
    Ky = trainable_parameters['Ky']
    Kz = trainable_parameters['Kz']
    Vd = trainable_parameters['Vd']
    Vm2 = trainable_parameters['Vm2']
    Vm3 = trainable_parameters['Vm3']
    Vp = trainable_parameters['Vp']
    beta_par = trainable_parameters['beta_par']
    epsilon_par = trainable_parameters['epsilon_par']
    n_par = trainable_parameters['n_par']
    v0 = trainable_parameters['v0']
    v1 = trainable_parameters['v1']
    offset = trainable_parameters['offset']
    scale = trainable_parameters['scale']

    Z = y[0]
    Y = y[1]
    A = y[2]

    dZdt = v0 + beta_par * v1 - Vm2 * Z ** 2 / (K2 ** 2 + Z ** 2) + Vm3 * A ** 4 * Y ** 2 * Z ** 4 / ((A ** 4 + Ka ** 4) * (Ky ** 2 + Y ** 2) * (Kz ** 4 + Z ** 4)) + Kf * Y - K_par * Z
    dYdt = Vm2 * Z ** 2 / (K2 ** 2 + Z ** 2) - Vm3 * A ** 4 * Y ** 2 * Z ** 4 / ((A ** 4 + Ka ** 4) * (Ky ** 2 + Y ** 2) * (Kz ** 4 + Z ** 4)) - Kf * Y
    dAdt = Vp * beta_par - Vd * A ** 2 * Z ** n_par / ((Kd ** n_par + Z ** n_par) * (A ** 2 + Kp ** 2)) - epsilon_par * A
    return np.array([dZdt, dYdt, dAdt])

def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    loss = 0.0
    loss += np.mean(np.square((solution[:, 0] - dataset[:, 0]) / (np.max(np.abs(dataset[:, 0])) + 1e-12)))
    loss = np.sqrt(loss / 1)
    return float(loss)

def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    writeout_array = np.zeros([solution_time.shape[0], 6])
    writeout_array[:, 0] = solution_time
    writeout_array[:, 1] = dataset[:, 0]
    writeout_array[:, 2] = solution[:, 0]
    writeout_array[:, 3] = solution[:, 1]
    writeout_array[:, 4] = solution[:, 2]
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    writeout_array[:, 5] = observables['Ca']
    return writeout_array

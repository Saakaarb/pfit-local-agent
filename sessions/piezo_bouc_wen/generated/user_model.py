import numpy as np



def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    alpha = trainable_parameters['alpha']
    beta = trainable_parameters['beta']
    gamma = trainable_parameters['gamma']
    cp = trainable_parameters['cp']
    kp = trainable_parameters['kp']
    de = trainable_parameters['de']
    mp = fixed_parameters['mp']
    xp = y[0]
    vp = y[1]
    h = y[2]
    dxpdt = vp
    dvpdt = (kp * (de * (24 + 24 * np.sin(16 * np.pi * t)) - h) - cp * vp - kp * xp) / mp
    dhdt = alpha * de * (24 * 16 * np.pi * np.cos(16 * np.pi * t)) - beta * np.abs(24 * 16 * np.pi * np.cos(16 * np.pi * t)) * np.abs(h) - gamma * (24 * 16 * np.pi * np.cos(16 * np.pi * t)) * np.abs(h)
    return np.array([dxpdt, dvpdt, dhdt])

def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    loss = 0.0
    loss += np.mean(np.square((solution[:, 0] - dataset[:, 0]) / (np.max(dataset[:, 0]) - np.min(dataset[:, 0]) + 1e-12)))
    return float(loss)

def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    writeout_array = np.zeros([solution_time.shape[0], 5])
    writeout_array[:, 0] = solution_time
    writeout_array[:, 1] = dataset[:, 0]
    writeout_array[:, 2] = solution[:, 0]
    writeout_array[:, 3] = solution[:, 1]
    writeout_array[:, 4] = solution[:, 2]
    return writeout_array

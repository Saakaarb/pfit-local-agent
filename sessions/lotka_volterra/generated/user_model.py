import numpy as np



def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    alpha = trainable_parameters['alpha']
    beta = trainable_parameters['beta']
    delta = trainable_parameters['delta']
    gamma = trainable_parameters['gamma']

    H = y[0]
    L = y[1]
    dHdt = alpha * H - beta * H * L
    dLdt = delta * H * L - gamma * L
    return np.array([dHdt, dLdt])

def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    loss = 0.0
    loss += np.mean(np.square((solution[:, 0] - dataset[:, 0]) / (np.max(dataset[:, 0]) - np.min(dataset[:, 0]) + 1e-12)))
    loss += np.mean(np.square((solution[:, 1] - dataset[:, 1]) / (np.max(dataset[:, 1]) - np.min(dataset[:, 1]) + 1e-12)))
    return float(loss)

def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    writeout_array = np.zeros([solution_time.shape[0], 5])
    writeout_array[:, 0] = solution_time
    writeout_array[:, 1] = dataset[:, 0]
    writeout_array[:, 2] = dataset[:, 1]
    writeout_array[:, 3] = solution[:, 0]
    writeout_array[:, 4] = solution[:, 1]
    return writeout_array

import numpy as np



def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    eps1 = trainable_parameters['eps1']
    eps2 = trainable_parameters['eps2']
    q = trainable_parameters['q']
    f = trainable_parameters['f']

    X = y[0]
    Y = y[1]
    Z = y[2]
    dXdt = (q * Y - X * Y + X * (1 - X)) / eps1
    dYdt = (-q * Y - X * Y + f * Z) / eps2
    dZdt = X - Z
    return np.array([dXdt, dYdt, dZdt])

def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    loss = 0.0
    loss += np.mean(np.square((solution[:, 0] - dataset[:, 0]) / (dataset[:, 2] + 1e-12)))
    loss += np.mean(np.square((solution[:, 2] - dataset[:, 1]) / (dataset[:, 3] + 1e-12)))
    return float(loss)

def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    writeout_array = np.zeros([solution_time.shape[0], 6])
    writeout_array[:, 0] = solution_time
    writeout_array[:, 1] = dataset[:, 0]
    writeout_array[:, 2] = dataset[:, 1]
    writeout_array[:, 3] = solution[:, 0]
    writeout_array[:, 4] = solution[:, 1]
    writeout_array[:, 5] = solution[:, 2]
    return writeout_array

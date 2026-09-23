import numpy as np



def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    k1 = trainable_parameters['k1']
    k2 = trainable_parameters['k2']
    k3 = trainable_parameters['k3']

    y1 = y[0]
    y2 = y[1]
    y3 = y[2]
    dy1dt = -k1 * y1 + k3 * y3 * y2
    dy2dt = k1 * y1 - k2 * y2 ** 2 - k3 * y2 * y3
    dy3dt = k2 * y2 ** 2
    return np.array([dy1dt, dy2dt, dy3dt])

def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    loss = 0.0
    loss += np.mean(np.square((solution[:, 0] - dataset[:, 0]) / (np.max(dataset[:, 0]) - np.min(dataset[:, 0]) + 1e-12)))
    loss += np.mean(np.square((solution[:, 1] - dataset[:, 1]) / (np.max(dataset[:, 1]) - np.min(dataset[:, 1]) + 1e-12)))
    eps_y3 = np.min(np.where(dataset[:, 2] > 0.0, dataset[:, 2], np.inf))
    log_sim_y3 = np.log10(solution[:, 2] + eps_y3)
    log_measured_y3 = np.log10(dataset[:, 2] + eps_y3)
    scale_log_y3 = np.max(log_measured_y3) - np.min(log_measured_y3) + 1e-12
    loss += np.mean(np.square((log_sim_y3 - log_measured_y3) / scale_log_y3))
    return float(loss)

def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    writeout_array = np.zeros([solution_time.shape[0], 7])
    writeout_array[:, 0] = solution_time
    writeout_array[:, 1] = dataset[:, 0]
    writeout_array[:, 2] = dataset[:, 1]
    writeout_array[:, 3] = dataset[:, 2]
    writeout_array[:, 4] = solution[:, 0]
    writeout_array[:, 5] = solution[:, 1]
    writeout_array[:, 6] = solution[:, 2]
    return writeout_array

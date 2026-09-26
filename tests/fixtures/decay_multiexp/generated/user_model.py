import numpy as np

def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    k1 = trainable_parameters['k1']
    k2 = trainable_parameters['k2']
    A, B = y
    return np.array([-k1 * A, k1 * A - k2 * B])

def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    scale_factor = np.max(np.abs(dataset), axis=0)
    scale_factor = np.where(scale_factor == 0.0, 1.0, scale_factor)
    return np.sqrt(np.mean(np.square((solution[:, 0:2] - dataset[:, 0:2]) / scale_factor[0:2])))

def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    return np.column_stack([solution_time, dataset[:, 0], dataset[:, 1], solution[:, 0], solution[:, 1]])

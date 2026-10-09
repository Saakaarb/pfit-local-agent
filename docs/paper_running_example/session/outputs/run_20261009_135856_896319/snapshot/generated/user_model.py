import numpy as np

def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    k_elim = trainable_parameters['k_elim']
    k_on = trainable_parameters['k_on']
    k_off = trainable_parameters['k_off']
    D = y[0]
    R = y[1]
    dose_rate = np.interp(t, t_eval, dataset[:, 0])
    return np.array([dose_rate - k_elim * D, k_on * D * (1.0 - R) - k_off * R])

def _observables(solution, trainable_parameters, fixed_parameters):
    return {'signal_model': trainable_parameters['baseline'] + trainable_parameters['gain'] * solution[:, 1]}

def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    signal_model = trainable_parameters['baseline'] + trainable_parameters['gain'] * solution[:, 1]
    signal = dataset[:, 1]
    mask = np.isfinite(signal)
    scale = np.max(np.abs(signal[mask]))
    residual = (signal_model[mask] - signal[mask]) / scale
    return float(np.sqrt(np.mean(residual * residual)))

def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    signal_model = trainable_parameters['baseline'] + trainable_parameters['gain'] * solution[:, 1]
    writeout_array = np.zeros([solution_time.shape[0], 5])
    writeout_array[:, 0] = solution_time
    writeout_array[:, 1] = solution[:, 0]
    writeout_array[:, 2] = solution[:, 1]
    writeout_array[:, 3] = dataset[:, 1]
    writeout_array[:, 4] = signal_model
    return writeout_array

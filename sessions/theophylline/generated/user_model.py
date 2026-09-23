import numpy as np

def _observables(solution, trainable_parameters, fixed_parameters):
    ka = trainable_parameters['ka']
    ke = trainable_parameters['ke']
    Vd = trainable_parameters['Vd']

    A_gut = solution[:, 0]
    A_plasma = solution[:, 1]
    return {
        'C_plasma': A_plasma / Vd,
    }


def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    ka = trainable_parameters['ka']
    ke = trainable_parameters['ke']
    Vd = trainable_parameters['Vd']

    A_gut = y[0]
    A_plasma = y[1]
    dA_gutdt = -ka * A_gut
    dA_plasmadt = ka * A_gut - ke * A_plasma
    return np.array([dA_gutdt, dA_plasmadt])

def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    loss = 0.0
    loss += np.mean(np.square((observables['C_plasma'] - dataset[:, 0]) / (np.max(dataset[:, 0]) - np.min(dataset[:, 0]) + 1e-12)))
    return float(loss)

def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    writeout_array = np.zeros([solution_time.shape[0], 5])
    writeout_array[:, 0] = solution_time
    writeout_array[:, 1] = dataset[:, 0]
    writeout_array[:, 2] = solution[:, 0]
    writeout_array[:, 3] = solution[:, 1]
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    writeout_array[:, 4] = observables['C_plasma']
    return writeout_array

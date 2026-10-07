import numpy as np

def _observables(solution, trainable_parameters, fixed_parameters):
    ka1 = trainable_parameters['ka1']
    ka2fold = trainable_parameters['ka2fold']
    kd1 = trainable_parameters['kd1']
    kd2fold = trainable_parameters['kd2fold']
    kin = trainable_parameters['kin']
    kin2 = trainable_parameters['kin2']
    kout = trainable_parameters['kout']
    kout2 = trainable_parameters['kout2']
    kout_frag = trainable_parameters['kout_frag']
    kon_unspec = trainable_parameters['kon_unspec']
    koff_unspec = trainable_parameters['koff_unspec']
    offset = trainable_parameters['offset']
    km = trainable_parameters['km']
    scale = trainable_parameters['scale']
    fragments = trainable_parameters['fragments']

    Ins = solution[:, 0]
    Rec1 = solution[:, 1]
    Rec2 = solution[:, 2]
    IR1 = solution[:, 3]
    IR2 = solution[:, 4]
    IR1in = solution[:, 5]
    IR2in = solution[:, 6]
    Uptake1 = solution[:, 7]
    Uptake2 = solution[:, 8]
    InsulinFragments = solution[:, 9]
    BoundUnspec = solution[:, 10]
    return {
        'Insulin_signal': offset + scale * (Ins + fragments * InsulinFragments) / (1 + (Ins + fragments * InsulinFragments) / km),
    }


def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    ka1 = trainable_parameters['ka1']
    ka2fold = trainable_parameters['ka2fold']
    kd1 = trainable_parameters['kd1']
    kd2fold = trainable_parameters['kd2fold']
    kin = trainable_parameters['kin']
    kin2 = trainable_parameters['kin2']
    kout = trainable_parameters['kout']
    kout2 = trainable_parameters['kout2']
    kout_frag = trainable_parameters['kout_frag']
    kon_unspec = trainable_parameters['kon_unspec']
    koff_unspec = trainable_parameters['koff_unspec']
    offset = trainable_parameters['offset']
    km = trainable_parameters['km']
    scale = trainable_parameters['scale']
    fragments = trainable_parameters['fragments']

    Ins = y[0]
    Rec1 = y[1]
    Rec2 = y[2]
    IR1 = y[3]
    IR2 = y[4]
    IR1in = y[5]
    IR2in = y[6]
    Uptake1 = y[7]
    Uptake2 = y[8]
    InsulinFragments = y[9]
    BoundUnspec = y[10]

    dInsdt = -(ka1 * Ins * Rec1) - ka1 * ka2fold * Ins * Rec2 - kon_unspec * Ins + koff_unspec * BoundUnspec + kd1 * IR1 + kd1 * kd2fold * IR2
    dRec1dt = -(ka1 * Ins * Rec1) + kd1 * IR1 + kout_frag * IR1in
    dRec2dt = -(ka1 * ka2fold * Ins * Rec2) + kd1 * kd2fold * IR2 + kout_frag * IR2in
    dIR1dt = ka1 * Ins * Rec1 - kd1 * IR1 - kin * IR1 + kout * IR1in
    dIR2dt = ka1 * ka2fold * Ins * Rec2 - kd1 * kd2fold * IR2 - kin2 * IR2 + kout2 * IR2in
    dIR1indt = kin * IR1 - kout * IR1in - kout_frag * IR1in
    dIR2indt = kin2 * IR2 - kout2 * IR2in - kout_frag * IR2in
    dUptake1dt = ka1 * Ins * Rec1 - kd1 * IR1
    dUptake2dt = ka1 * ka2fold * Ins * Rec2 - kd1 * kd2fold * IR2
    dInsulinFragmentsdt = kout_frag * IR1in + kout_frag * IR2in
    dBoundUnspecdt = kon_unspec * Ins - koff_unspec * BoundUnspec
    return np.array([dInsdt, dRec1dt, dRec2dt, dIR1dt, dIR2dt, dIR1indt, dIR2indt, dUptake1dt, dUptake2dt, dInsulinFragmentsdt, dBoundUnspecdt])

def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    loss = 0.0
    loss += np.mean(np.square(observables['Insulin_signal'] - dataset[:, 0]))
    return float(loss)

def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    writeout_array = np.zeros([solution_time.shape[0], 14])
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
    writeout_array[:, 10] = solution[:, 8]
    writeout_array[:, 11] = solution[:, 9]
    writeout_array[:, 12] = solution[:, 10]
    observables = _observables(solution, trainable_parameters, fixed_parameters)
    writeout_array[:, 13] = observables['Insulin_signal']
    return writeout_array

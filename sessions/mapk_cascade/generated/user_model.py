import numpy as np



def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    k1 = trainable_parameters['k1']
    k_1 = trainable_parameters['k_1']
    k2 = trainable_parameters['k2']
    k3 = trainable_parameters['k3']
    k_3 = trainable_parameters['k_3']
    k4 = trainable_parameters['k4']
    h1 = trainable_parameters['h1']
    h_1 = trainable_parameters['h_1']
    h2 = trainable_parameters['h2']
    h3 = trainable_parameters['h3']
    h_3 = trainable_parameters['h_3']
    h4 = trainable_parameters['h4']
    h_4 = trainable_parameters['h_4']
    h5 = trainable_parameters['h5']
    h6 = trainable_parameters['h6']
    h_6 = trainable_parameters['h_6']

    M = y[0]
    Mp = y[1]
    Mpp = y[2]
    MAPKK = y[3]
    MKP3 = y[4]
    M_MAPKK = y[5]
    Mp_MAPKK = y[6]
    Mpp_MKP3 = y[7]
    Mp_MKP3_dep = y[8]
    Mp_MKP3 = y[9]
    M_MKP3 = y[10]
    dMdt = -(k1 * M * MAPKK - k_1 * M_MAPKK) + (h6 * M_MKP3 - h_6 * M * MKP3)
    dMpdt = k2 * M_MAPKK - (k3 * Mp * MAPKK - k_3 * Mp_MAPKK) + (h3 * Mp_MKP3_dep - h_3 * Mp * MKP3) - (h4 * Mp * MKP3 - h_4 * Mp_MKP3)
    dMppdt = k4 * Mp_MAPKK - (h1 * Mpp * MKP3 - h_1 * Mpp_MKP3)
    dMAPKKdt = -(k1 * M * MAPKK - k_1 * M_MAPKK) + k2 * M_MAPKK - (k3 * Mp * MAPKK - k_3 * Mp_MAPKK) + k4 * Mp_MAPKK
    dMKP3dt = -(h1 * Mpp * MKP3 - h_1 * Mpp_MKP3) + (h3 * Mp_MKP3_dep - h_3 * Mp * MKP3) - (h4 * Mp * MKP3 - h_4 * Mp_MKP3) + (h6 * M_MKP3 - h_6 * M * MKP3)
    dM_MAPKKdt = k1 * M * MAPKK - k_1 * M_MAPKK - k2 * M_MAPKK
    dMp_MAPKKdt = k3 * Mp * MAPKK - k_3 * Mp_MAPKK - k4 * Mp_MAPKK
    dMpp_MKP3dt = h1 * Mpp * MKP3 - h_1 * Mpp_MKP3 - h2 * Mpp_MKP3
    dMp_MKP3_depdt = h2 * Mpp_MKP3 - (h3 * Mp_MKP3_dep - h_3 * Mp * MKP3)
    dMp_MKP3dt = h4 * Mp * MKP3 - h_4 * Mp_MKP3 - h5 * Mp_MKP3
    dM_MKP3dt = h5 * Mp_MKP3 - (h6 * M_MKP3 - h_6 * M * MKP3)
    return np.array([dMdt, dMpdt, dMppdt, dMAPKKdt, dMKP3dt, dM_MAPKKdt, dMp_MAPKKdt, dMpp_MKP3dt, dMp_MKP3_depdt, dMp_MKP3dt, dM_MKP3dt])

def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    loss = 0.0
    loss += np.mean(np.square((solution[:, 2] - dataset[:, 0]) / (np.max(dataset[:, 0]) - np.min(dataset[:, 0]) + 1e-12)))
    return float(loss)

def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    writeout_array = np.zeros([solution_time.shape[0], 13])
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
    return writeout_array

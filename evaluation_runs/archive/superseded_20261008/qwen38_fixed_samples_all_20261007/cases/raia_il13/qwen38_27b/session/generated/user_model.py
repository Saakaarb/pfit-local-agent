import numpy as np



def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    Kon_IL13Rec = trainable_parameters['Kon_IL13Rec']
    Rec_intern = trainable_parameters['Rec_intern']
    Rec_recycle = trainable_parameters['Rec_recycle']
    Rec_phosphorylation = trainable_parameters['Rec_phosphorylation']
    JAK2_phosphorylation = trainable_parameters['JAK2_phosphorylation']
    JAK2_p_inhibition = trainable_parameters['JAK2_p_inhibition']
    pRec_intern = trainable_parameters['pRec_intern']
    pRec_degradation = trainable_parameters['pRec_degradation']
    pJAK2_dephosphorylation = trainable_parameters['pJAK2_dephosphorylation']
    STAT5_phosphorylation = trainable_parameters['STAT5_phosphorylation']
    pSTAT5_dephosphorylation = trainable_parameters['pSTAT5_dephosphorylation']
    SOCS3mRNA_production = trainable_parameters['SOCS3mRNA_production']
    SOCS3_translation = trainable_parameters['SOCS3_translation']
    SOCS3_accumulation = trainable_parameters['SOCS3_accumulation']
    SOCS3_degradation = trainable_parameters['SOCS3_degradation']
    DecoyR_binding = fixed_parameters['DecoyR_binding']
    CD274mRNA_production = fixed_parameters['CD274mRNA_production']
    Rec = y[0]
    Rec_i = y[1]
    IL13_Rec = y[2]
    p_IL13_Rec = y[3]
    p_IL13_Rec_i = y[4]
    JAK2 = y[5]
    pJAK2 = y[6]
    STAT5 = y[7]
    pSTAT5 = y[8]
    SOCS3mRNA = y[9]
    DecoyR = y[10]
    IL13_DecoyR = y[11]
    SOCS3 = y[12]
    CD274mRNA = y[13]
    il13_level = y[14]

    dRecdt = -(2.265 * il13_level * Kon_IL13Rec * Rec) - Rec * Rec_intern + Rec_i * Rec_recycle
    dRec_idt = Rec * Rec_intern - Rec_i * Rec_recycle
    dIL13_Recdt = 2.265 * il13_level * Kon_IL13Rec * Rec - IL13_Rec * Rec_phosphorylation * pJAK2
    dp_IL13_Recdt = IL13_Rec * Rec_phosphorylation * pJAK2 - pRec_intern * p_IL13_Rec
    dp_IL13_Rec_idt = pRec_intern * p_IL13_Rec - pRec_degradation * p_IL13_Rec_i
    dJAK2dt = -(IL13_Rec * JAK2 * JAK2_phosphorylation / (1 + JAK2_p_inhibition * SOCS3)) - JAK2 * JAK2_phosphorylation * p_IL13_Rec / (1 + JAK2_p_inhibition * SOCS3) + 91 * pJAK2 * pJAK2_dephosphorylation
    dpJAK2dt = IL13_Rec * JAK2 * JAK2_phosphorylation / (1 + JAK2_p_inhibition * SOCS3) + JAK2 * JAK2_phosphorylation * p_IL13_Rec / (1 + JAK2_p_inhibition * SOCS3) - 91 * pJAK2 * pJAK2_dephosphorylation
    dSTAT5dt = -(STAT5 * STAT5_phosphorylation * pJAK2) + 91 * pSTAT5 * pSTAT5_dephosphorylation
    dpSTAT5dt = STAT5 * STAT5_phosphorylation * pJAK2 - 91 * pSTAT5 * pSTAT5_dephosphorylation
    dSOCS3mRNAdt = SOCS3mRNA_production * pSTAT5
    dDecoyRdt = -(2.265 * il13_level * DecoyR * DecoyR_binding)
    dIL13_DecoyRdt = 2.265 * il13_level * DecoyR * DecoyR_binding
    dSOCS3dt = SOCS3mRNA * SOCS3_translation / (SOCS3mRNA + SOCS3_accumulation) - SOCS3 * SOCS3_degradation
    dCD274mRNAdt = CD274mRNA_production * pSTAT5
    dil13_leveldt = 0
    return np.array([dRecdt, dRec_idt, dIL13_Recdt, dp_IL13_Recdt, dp_IL13_Rec_idt, dJAK2dt, dpJAK2dt, dSTAT5dt, dpSTAT5dt, dSOCS3mRNAdt, dDecoyRdt, dIL13_DecoyRdt, dSOCS3dt, dCD274mRNAdt, dil13_leveldt])

def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    loss = 0.0
    loss += np.mean(np.square((solution[:, 8] - dataset[:, 0]) / (np.max(np.abs(dataset[:, 0])) + 1e-12)))
    loss = np.sqrt(loss / 1)
    return float(loss)

def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    writeout_array = np.zeros([solution_time.shape[0], 17])
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
    writeout_array[:, 13] = solution[:, 11]
    writeout_array[:, 14] = solution[:, 12]
    writeout_array[:, 15] = solution[:, 13]
    writeout_array[:, 16] = solution[:, 14]
    return writeout_array

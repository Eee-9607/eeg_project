# -*- coding = utf-8 -*-
# @Time : 19/01/2024 11.08
# @Author : yangjw
# @Site : 
# @File : bhv_function.py
# @Software : PyCharm
# @contact: jwyang9826@gmail.com

import numpy as np
import pandas as pd

def find_useless_trial(event_mat, stim_info):
    mat_var = event_mat['save_variable']
    keys_list = mat_var.dtype.fields.keys()

    event_df = pd.DataFrame([])
    for key_index, key in enumerate(keys_list):
        trial_num = len(mat_var[key].squeeze())

        row_array = np.array([mat_var[key][0, trial][0][0]
                              for trial in np.arange(trial_num)]).reshape([trial_num, 1])

        # change the columns' name because of recording mistake
        if key == 'ip':
            row_df = pd.DataFrame(row_array, columns=['dp'])
        elif key == 'dp':
            row_df = pd.DataFrame(row_array, columns=['ip'])
        else:
            row_df = pd.DataFrame(row_array, columns=[key])

        event_df = pd.concat((event_df, row_df), axis=1)

    ges_index = []
    for e_index in event_df.T:
        if event_df.loc[e_index, 'dp'] in list(stim_info.loc[:, 'ges_dp']):
            for stim_index in stim_info.T:
                if event_df.loc[e_index, 'dp'] == stim_info.loc[stim_index, 'ges_dp']:
                    ges_index.append(stim_index)
        else:
            ges_index.append(999)

    spe_index = []
    for e_index in event_df.T:
        if event_df.loc[e_index, 'ip'] in list(stim_info.loc[:, 'spe_ip']):
            for stim_index in stim_info.T:
                if event_df.loc[e_index, 'ip'] == stim_info.loc[stim_index, 'spe_ip']:
                    if event_df.loc[e_index, 'ip'] == 263:
                        if event_df.loc[e_index, 'pair'] == 7:
                            spe_index.append(4)
                            break
                        elif event_df.loc[e_index, 'pair'] == 11:
                            spe_index.append(11)
                            break
                    else:
                        spe_index.append(stim_index)
        else:
            spe_index.append(999)

    event_df['dp_stim'] = np.array(ges_index)
    event_df['ip_stim'] = np.array(spe_index)
    delete_id = [i_index for i_index, i in enumerate(ges_index) if i == 999]
    new_event_df = event_df.drop(delete_id, axis=0)

    return delete_id, new_event_df


def cal_entropy(*args, **kwargs):

    k = kwargs

    if k['entropy_opt'] == 'single':
        x = args[0]
        h = 0

        if np.sum(x) != 1:
            Warning('The sum of possibility is ' + str(np.sum(x)))

        for cell_pos in x:
            single_e = -cell_pos * np.log2(cell_pos)
            h += single_e

        return h

    elif k['entropy_opt'] == 'joint':

        x = args[0]
        y = args[1]
        joint_dis = np.zeros([len(x) * len(y)])

        index = 0
        for single_x in x:
            for single_y in y:
                joint_dis[index] = single_x * single_y
                index += 1

        if np.sum(x) != 1 or np.sum(y) != 1 or np.sum(joint_dis) != 1:
            Warning('The sum of x_possibility is ' + str(np.sum(x)) + '\n' +
                    'The sum of y_possibility is ' + str(np.sum(y)) + '\n' +
                    'The sum of joint_possibility is ' + str(np.sum(joint_dis)))

        h = cal_entropy(joint_dis, entropy_opt='single')

        return h, joint_dis

    elif k['entropy_opt'] == 'mutual':
        x = args[0]
        y = args[1]
        joint_dis = np.zeros([len(x) * len(y)])

        index = 0
        for single_x in x:
            for single_y in y:
                joint_dis[index] = single_x * single_y
                index += 1

        if np.sum(x) != 1 or np.sum(y) != 1 or np.sum(joint_dis) != 1:
            Warning('The sum of x_possibility is ' + str(np.sum(x)) + '\n' +
                    'The sum of y_possibility is ' + str(np.sum(y)) + '\n' +
                    'The sum of joint_possibility is ' + str(np.sum(joint_dis)))


        joint_x = cal_entropy(x, entropy_opt='single')
        joint_y = cal_entropy(y, entropy_opt='single')
        joint_h = cal_entropy(joint_dis, entropy_opt='single')

        h = joint_x + joint_y - joint_h

        return h, joint_dis
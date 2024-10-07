# -*- coding = utf-8 -*-
# @Time : 07/10/2024 14.55
# @Author : yangjw
# @Site : 
# @File : calculating_multisensory_entropy.py
# @Software : PyCharm
# @contact: jwyang9826@gmail.com



from glob import glob
import numpy as np
import pandas as pd
from jw_function.bhv_function import cal_entropy
import os
from os.path import join as oj

work_directory = os.getcwd()
project_path = oj(work_directory, 'data', 'pretest_data', 'log_naming')

all_sub = np.arange(101, 131, 1)
miss_sub = [101]
sub_list = [sub for sub in all_sub if sub not in miss_sub]

all_cond_sc = {}
all_cond_si = {}
for cond_index, cond in enumerate(np.arange(1, 10)):
    for session_index, session in enumerate(np.arange(1, 3)):
        txt_paths = list(glob(f"{project_path}/*{session}_{cond}.txt"))
        txt_paths.sort()

        all_sc_sub_answer = pd.DataFrame([])
        all_si_sub_answer = pd.DataFrame([])
        for txt_index, txt in enumerate(txt_paths):

            sub_id = txt.split("\\")[-1].split('_')[0]
            session = txt.split("\\")[-1].split('_')[1]
            # cond = txt.split("\\")[-1].split('_')[2]

            if int(sub_id) not in miss_sub:

                with open(txt, 'r', encoding='utf-8') as f:
                    data = f.readlines()

                    sc_material_dict = {}
                    si_material_dict = {}

                    for row in np.arange(len(data) - 1):
                        single_str = data[row + 1]
                        str_tuple = single_str.split('\n')
                        new_str_tuple = str_tuple[0].split(' ')

                        if new_str_tuple[-2] == 'next':
                            target_row = data[row].split('\n')
                            new_target_row = target_row[0].split(' ')

                            if new_target_row[-2].isdigit():
                                answer = new_target_row[-1]
                            else:
                                answer = new_target_row[-2]

                            sc = new_target_row[6]

                            if sc == 'sc':
                                material = new_target_row[4].split('-')[1].split('.avi')[0]
                                sc_material_dict[material] = answer

                            elif sc == 'si':

                                if session == '1':
                                    material = new_target_row[4].split('-')[1].split('.avi')[0]
                                elif session == '2':
                                    material = new_target_row[5].split('-')[1].split('.wav')[0]

                                si_material_dict[material] = answer

                    f.close()

                    sc_material_df = pd.DataFrame(sc_material_dict, index=['sc-sub-' + sub_id])
                    si_material_df = pd.DataFrame(si_material_dict, index=['si-sub-' + sub_id])

                all_sc_sub_answer = pd.concat((all_sc_sub_answer, sc_material_df), axis=0)
                all_si_sub_answer = pd.concat((all_si_sub_answer, si_material_df), axis=0)

        all_cond_sc[session + '_' + str(cond)] = all_sc_sub_answer
        all_cond_si[session + '_' + str(cond)] = all_si_sub_answer

full_sc_answer_df = {}
full_si_answer_df = {}

for cond in np.arange(1, 10):
    for session_index, session in enumerate(np.arange(1, 3)):
        # this part we calculate the possibility distribution about sc condition
        cond_answer_df = all_cond_sc[str(session) + '_' + str(cond)].copy()
        full_answer_df = pd.DataFrame([])
        for item in cond_answer_df.columns:
            single_answer = cond_answer_df[item].dropna(axis=0).value_counts()
            single_df = pd.DataFrame(single_answer)

            probability = np.array([i/float(single_df.sum()['count']) for i in single_df.loc[:, 'count']])
            single_df['probability'] = probability
            single_df = single_df.reset_index()
            single_df.rename(columns={item: 'answer'}, inplace=True)

            single_df['stim'] = np.array([item + '.avi' for i in np.arange(single_df.shape[0])])

            full_answer_df = pd.concat((full_answer_df, single_df), axis=0)

        full_sc_answer_df[str(session) + '_' + str(cond)] = full_answer_df

        # this part we calculate the possibility distribution about si condition
        si_cond_answer_df = all_cond_si[str(session) + '_' + str(cond)].copy()
        full_si_df = pd.DataFrame([])
        for item in si_cond_answer_df.columns:
            single_answer = si_cond_answer_df[item].dropna(axis=0).value_counts()
            single_df = pd.DataFrame(single_answer)

            probability = np.array([i / float(single_df.sum()['count']) for i in single_df.loc[:, 'count']])
            single_df['probability'] = probability
            single_df = single_df.reset_index()
            single_df.rename(columns={item: 'answer'}, inplace=True)

            single_df['stim'] = np.array([item + '.avi' for i in np.arange(single_df.shape[0])])

            full_si_df = pd.concat((full_si_df, single_df), axis=0)

        full_si_answer_df[str(session) + '_' + str(cond)] = full_si_df


# calculate speech and gesture entropy
outside_path = oj(work_directory, 'data', 'pretest_data', 'first_version')
cond_xls = pd.read_csv(outside_path + '1.csv')
cond_xls = cond_xls.dropna(axis=0)

full_xls = cond_xls.T.reset_index().T.reset_index().drop(labels='index', axis=1)

new_xls = pd.DataFrame(np.zeros(full_xls.shape), dtype='int')
for col_index, col in enumerate(full_xls.columns):
    for row_index, row in enumerate(full_xls.index):
        single_col = full_xls.loc[row, col]

        if col_index not in [1, 2, 3, 4]:
            new_xls.iloc[row_index, col_index] = int(float(single_col))
        else:
            new_xls.iloc[row_index, col_index] = single_col

main_directory = oj(work_directory, 'entropy')
independent_gesture = pd.read_excel(main_directory + 'gesture_entropy_full_item.xlsx', index_col=0)
independent_speech = pd.read_excel(main_directory + 'speech_entropy_full_item.xlsx', index_col=0)

stimulus_list = np.array(pd.value_counts(np.array([stim_name.split('-')[1].split('.')[0]
                                                   for stim_name in new_xls.iloc[0:-2, 1]])).index)

gesture_entropy = pd.DataFrame([])
speech_entropy = pd.DataFrame([])
mutual_information = np.zeros([9, len(stimulus_list)])
combine_mi = np.zeros([9, len(stimulus_list)])

for cond in np.arange(1, 10):
    cond_sc_gesture = full_sc_answer_df['1_' + str(cond)].copy()
    cond_sc_speech = full_sc_answer_df['2_' + str(cond)].copy()
    gesture_dict = {}
    speech_dict = {}
    for item_index, item in enumerate(stimulus_list):
        left_stim = cond_sc_gesture.loc[[item == stim.split('.avi')[0] for stim in cond_sc_gesture['stim']], :]
        item_entropy = cal_entropy(left_stim)
        gesture_dict[item] = item_entropy

        # now we calculate speech entropy
        left_speech_stim = cond_sc_speech.loc[[item == stim.split('.avi')[0] for stim in cond_sc_speech['stim']], :]
        speech_item_entropy = cal_entropy(left_speech_stim)
        speech_dict[item] = speech_item_entropy

    cond_gesture_df = pd.DataFrame([gesture_dict], index=[cond])
    gesture_entropy = pd.concat((gesture_entropy, cond_gesture_df), axis=0)
    cond_speech_df = pd.DataFrame([speech_dict], index=[cond])
    speech_entropy = pd.concat((speech_entropy, cond_speech_df), axis=0)

co_gesture = np.average(gesture_entropy.mean(axis=1))
co_speech = np.average(speech_entropy.mean(axis=1))

single_ges = np.average(independent_gesture.mean(axis=1))
single_spe = np.average(independent_speech.mean(axis=1))


dp_name = ['BD', 'DP', 'AD']
ip_name = ['BI', 'IP', 'AI']
all_cond_name = np.array([dp + '&' + ip for ip in ip_name for dp in dp_name])

gesture_entropy.index = pd.Index(all_cond_name)
speech_entropy.index = pd.Index(all_cond_name)

entropy_path = oj(work_directory, 'entropy')
gesture_entropy.to_excel(entropy_path + 'new_co_gesture.xlsx')
speech_entropy.to_excel(entropy_path + 'new_co_speech.xlsx')

mean_independent_gesture = independent_gesture.iloc[:, 1::].mean(axis=0)
mean_independent_speech = independent_speech.iloc[:, 1::].mean(axis=0)

information_gain_ges = np.zeros([3, 3])
information_gain_spe = np.zeros([3, 3])
ges_gain_rate = np.zeros([3, 3])
spe_gain_rate = np.zeros([3, 3])

g_index_list = [(1, 4, 7), (2, 5, 8), (3, 6, 9)]
s_index_list = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]

information_gain_ges_df = np.zeros([3, 3, len(stimulus_list)])
information_gain_spe_df = np.zeros([3, 3, len(stimulus_list)])
information_gain_ges_rate = np.zeros([3, 3, len(stimulus_list)])
information_gain_spe_rate = np.zeros([3, 3, len(stimulus_list)])
independent_ges_df = independent_gesture.set_index(independent_gesture.index)
independent_spe_df = independent_speech.set_index(independent_speech.index)

length_index = [0.75, 1, 1.25]

for g_i, g_index in enumerate(g_index_list):
    for stim_index, stim_name in enumerate(stimulus_list):
        information_gain_ges_df[g_i, :, stim_index] = independent_ges_df.loc[stim_name, length_index[g_i]] - \
                                                 gesture_entropy.loc[all_cond_name[np.array(g_index) - 1], stim_name]
        information_gain_ges_rate[g_i, :, stim_index] = (independent_ges_df.loc[stim_name, length_index[g_i]] -
                                                         gesture_entropy.loc[
                                                             all_cond_name[np.array(g_index) - 1],  stim_name]) / \
                                                         independent_ges_df.loc[stim_name, length_index[g_i]]

for s_i, s_index in enumerate(s_index_list):
    for stim_index, stim_name in enumerate(stimulus_list):
        information_gain_spe_df[s_i, :, stim_index] = independent_spe_df.loc[stim_name, length_index[s_i]] - \
                                                 speech_entropy.loc[all_cond_name[np.array(s_index) - 1], stim_name]
        information_gain_spe_rate[s_i, :, stim_index] = (independent_spe_df.loc[stim_name, length_index[s_i]] -
                                                         gesture_entropy.loc[
                                                             all_cond_name[np.array(s_index) - 1], stim_name]) / \
                                                        independent_spe_df.loc[stim_name, length_index[s_i]]


dp_cond_name = [dpn + '&' + ipn for dpn in dp_name for ipn in ip_name]
ip_cond_name = [dpn + '&' + ipn for ipn in ip_name for dpn in dp_name]

flatten_information_gain_ges = pd.DataFrame(information_gain_ges_df.reshape([9, 19]).T, index=stimulus_list,
                                            columns=dp_cond_name)
flatten_information_gain_spe = pd.DataFrame(information_gain_spe_df.reshape([9, 19]).T, index=stimulus_list,
                                            columns=ip_cond_name)

flatten_information_gain_spe.to_excel(entropy_path + 'new_information_gain_spe_check.xlsx')
flatten_information_gain_ges.to_excel(entropy_path + 'new_information_gain_ges_check.xlsx')

# generate two null excel
null_excel = pd.DataFrame([])
null_excel.to_excel(entropy_path + 'information_gain_ges.xlsx')
null_excel.to_excel(entropy_path + 'information_gain_spe.xlsx')
ges_writer = pd.ExcelWriter(entropy_path + 'information_gain_ges.xlsx')
spe_writer = pd.ExcelWriter(entropy_path + 'information_gain_spe.xlsx')


for stim_index, stim_name in enumerate(stimulus_list):
    ges_cond = pd.DataFrame(information_gain_ges_df[:, :, stim_index], index=dp_name, columns=ip_name)
    ges_cond.to_excel(ges_writer, sheet_name=stim_name)

    spe_cond = pd.DataFrame(information_gain_spe_df[:, :, stim_index], index=ip_name, columns=dp_name)
    spe_cond.to_excel(spe_writer, sheet_name=stim_name)


ges_writer.close()
spe_writer.close()


for g_i, g_index in enumerate(g_index_list):
    information_gain_ges[g_i, :] = mean_independent_gesture[g_i] - co_gesture.iloc[np.array(g_index) - 1]
    ges_gain_rate[g_i, :] = (mean_independent_gesture[g_i] -
                             co_gesture.iloc[np.array(g_index) - 1])/mean_independent_gesture[g_i]

for s_i, s_index in enumerate(s_index_list):
    information_gain_spe[s_i, :] = (mean_independent_speech[s_i] - co_speech.iloc[np.array(s_index) - 1])
    spe_gain_rate[s_i, :] = (mean_independent_speech[s_i] -
                             co_speech.iloc[np.array(s_index) - 1]) / mean_independent_speech[s_i]

information_spe_df = pd.DataFrame(information_gain_spe, index=['before IP', 'IP', 'after IP'],
                                  columns=['before DP', 'DP', 'after DP'])
information_ges_df = pd.DataFrame(information_gain_ges, index=['before DP', 'DP', 'after DP'],
                                  columns=['before IP', 'IP', 'after IP'])
spe_gain_rate_df = pd.DataFrame(spe_gain_rate, index=['before IP', 'IP', 'after IP'],
                                columns=['before DP', 'DP', 'after DP'])
ges_gain_rate_df = pd.DataFrame(ges_gain_rate, index=['before DP', 'DP', 'after DP'],
                                columns=['before IP', 'IP', 'after IP'])

# ok, next part is calculating mutual information
all_combine = {}
for cond in np.arange(1, 10):
    cond_sc_gesture_answer = all_cond_sc['1_' + str(cond)].copy()
    cond_sc_speech_answer = all_cond_sc['2_' + str(cond)].copy()

    cond_combine = pd.DataFrame([])
    for sub_index, sub in enumerate(sub_list):
        sub_gesture = cond_sc_gesture_answer.loc['sc-sub-' + str(sub), :]
        sub_speech = cond_sc_speech_answer.loc['sc-sub-' + str(sub), :]

        sub_combine = pd.concat((sub_gesture, sub_speech), axis=1)

        new_combine_answer = [str(sub_combine.iloc[cm_row, 0]) + '&' + str(sub_combine.iloc[cm_row, 1])
                              for cm_row in np.arange(sub_combine.shape[0])]
        new_combine_df = pd.DataFrame(new_combine_answer, index=sub_combine.index, columns=['sc-sub-' + str(sub)])
        cond_combine = pd.concat((cond_combine, new_combine_df.T), axis=0)

    all_combine[cond] = cond_combine

# constructing answer distribution
all_cond_combine_sc_dict = {}
for cond in np.arange(1, 10):
    cond_combine = all_combine[cond].copy()

    full_combine_sc_df = pd.DataFrame([])

    for item_index, item in enumerate(cond_combine.columns):
        stimulus_answer = cond_combine.loc[:, item]
        stimulus_nan_bool = ['nan' not in cm_answer for cm_answer in stimulus_answer]
        left_combine_stim = stimulus_answer.loc[stimulus_nan_bool]
        combine_answer_count = left_combine_stim.value_counts()

        combine_df = pd.DataFrame(combine_answer_count)
        probability = np.array([i / float(combine_df.sum()['count']) for i in combine_df.loc[:, 'count']])
        combine_df['probability'] = probability
        combine_df = combine_df.reset_index()
        combine_df.rename(columns={item: 'answer'}, inplace=True)

        combine_df['stim'] = np.array([item + '.avi' for i in np.arange(combine_df.shape[0])])

        full_combine_sc_df = pd.concat((full_combine_sc_df, combine_df), axis=0)

    all_cond_combine_sc_dict[cond] = full_combine_sc_df

combine_entropy = pd.DataFrame([])
for cond in np.arange(1, 10):
    cond_sc_combine = all_cond_combine_sc_dict[cond].copy()
    combine_dict = {}
    for item_index, item in enumerate(stimulus_list):
        left_cm_stim = cond_sc_combine.loc[[item == stim.split('.avi')[0] for stim in cond_sc_combine['stim']], :]
        item_entropy = cal_entropy(left_cm_stim)
        combine_dict[item] = item_entropy

        # now we calculate speech entropy

    cond_combine_df = pd.DataFrame([combine_dict], index=[cond])
    combine_entropy = pd.concat((combine_entropy, cond_combine_df), axis=0)

mutual_information = gesture_entropy + speech_entropy - combine_entropy

mean_mi = mutual_information.mean(axis=1)

nine_inde_gesture = np.zeros([9, 19])
nine_inde_speech = np.zeros([9, 19])

ges_index = [0, 1, 2, 0, 1, 2, 0, 1, 2]
spe_index = [0, 0, 0, 1, 1, 1, 2, 2, 2]

for dp_cond in np.arange(9):
    nine_inde_gesture[dp_cond, :] = independent_gesture.iloc[0:19, ges_index[dp_cond] + 1]
    nine_inde_speech[dp_cond, :] = independent_speech.iloc[0:19, spe_index[dp_cond] + 1]

nine_inde_ges_df = pd.DataFrame(nine_inde_gesture, columns=pd.Index(independent_gesture.iloc[0:19, 0]))
nine_inde_spe_df = pd.DataFrame(nine_inde_speech, columns=pd.Index(independent_speech.iloc[0:19, 0]))
common_columns = combine_entropy.columns.intersection(nine_inde_ges_df.columns)

nine_ges_df = nine_inde_ges_df.reindex(columns=common_columns)
nine_spe_df = nine_inde_spe_df.reindex(columns=common_columns)
nine_ges_df.index = pd.Index(np.arange(1, 10))
nine_spe_df.index = pd.Index(np.arange(1, 10))

independent_mi = nine_ges_df + nine_spe_df - combine_entropy
mean_in_mi = independent_mi.mean(axis=1)

dp_name = ['before DP', 'DP', 'after DP']
ip_name = ['before IP', 'IP', 'after IP']
mean_mi = pd.DataFrame(np.array(mean_mi).reshape([3, 3]), index=[ip_name], columns=[dp_name])

mean_in_mi = pd.DataFrame(np.array(mean_in_mi).reshape([3, 3]), index=[ip_name], columns=[dp_name])

gesture_entropy.to_excel(main_directory + 'gesture entropy in joint.xlsx')
speech_entropy.to_excel(main_directory + 'speech entropy in joint.xlsx')
combine_entropy.to_excel(main_directory + 'joint entropy in joint.xlsx')

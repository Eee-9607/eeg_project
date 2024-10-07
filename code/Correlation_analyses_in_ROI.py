# -*- coding = utf-8 -*-
# @Time : 30/09/2024 21.42
# @Author : yangjw
# @Site : 
# @File : Correlation_analyses_in_ROI.py
# @Software : PyCharm
# @contact: jwyang9826@gmail.com


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.io as sio
import os
import mne
from os.path import join as oj
import pingouin as pg
from sklearn.preprocessing import StandardScaler
from jw_function.bhv_function import find_useless_trial
from jw_function.supply_function import reshape_array


work_directory = os.getcwd()
fig_path = oj(work_directory, 'figure')

behavior_path = oj(work_directory, 'behavioral_information')

excel_path = oj(behavior_path, 'stimuli_information_update.xls')
stim_info = pd.read_excel(excel_path)

entropy_path = oj(work_directory, 'entropy')

g_entropy_path = oj(entropy_path, 'gesture_entropy_full_item.xlsx')
g_entropy = pd.read_excel(g_entropy_path, index_col=0)
s_entropy_path = oj(entropy_path, 'speech_entropy_full_item.xlsx')
s_entropy = pd.read_excel(s_entropy_path, index_col=0)

entropy_loss = pd.read_excel(entropy_path + "/entropy_loss.xlsx", index_col=0)

item_rank = pd.read_excel(entropy_path + '/itemrank2.xls', header=None)

dp_cond = np.arange(1, 4)
ip_cond = np.arange(1, 4)
congru_cond = np.arange(1, 3)

all_sub = np.arange(1, 40)
miss_sub = [7, 8, 10, 17, 20, 21, 23, 24, 25]
sub_list = [sub for sub in all_sub if sub not in miss_sub]

dp_name = ['Before DP', 'DP', 'After DP']
ip_name = ['Before IP', 'IP', 'After IP']
all_cond_name = [f"{dpn} & {ipn}" for dpn in dp_name for ipn in ip_name]

dp_iname = ['BD', 'DP', 'AD']
ip_iname = ['BI', 'IP', 'AI']

index_cond_name = [f"{dpn}&{ipn}" for dpn in dp_iname for ipn in ip_iname]

stimulus_list = [sn.split('.avi')[0] for sn in stim_info.iloc[:, 0]]
chan_num = 42
time_point = 600
method = 'spearman'

mi_path = entropy_path + '/mutual_information_full_item.xlsx'
mi_entropy = pd.read_excel(mi_path, index_col=0)
mi_data = pd.DataFrame(np.array(mi_entropy.iloc[:, 1::]), index=stimulus_list, columns=index_cond_name)

ig_ges_path = entropy_path + '/information_gain_ges.xlsx'
ig_spe_path = entropy_path + '/information_gain_spe.xlsx'

ig_ges_array = np.zeros([9, 19])
ig_spe_array = np.zeros([9, 19])

cond_index = 0
for dp_index in np.arange(3):
    for ip_index in np.arange(3):
        for stim_index, stim_name in enumerate(stimulus_list):
            ig_ges_df = pd.read_excel(ig_ges_path, sheet_name=stim_name, index_col=0)
            ig_ges_array[cond_index, stim_index] = ig_ges_df.iloc[dp_index, ip_index]

            ig_spe_df = pd.read_excel(ig_spe_path, sheet_name=stim_name, index_col=0)
            ig_spe_array[cond_index, stim_index] = ig_spe_df.iloc[ip_index, dp_index]

        cond_index += 1

ig_ges_data = pd.DataFrame(ig_ges_array, index=index_cond_name, columns=stimulus_list).T
ig_spe_data = pd.DataFrame(ig_spe_array, index=index_cond_name, columns=stimulus_list).T

rt_data = np.array(pd.read_excel(work_directory + '/full rt differences.xlsx', index_col=0))

spe_cond_data = np.zeros([19, 9])
ges_cond_data = np.zeros([19, 9])
spe_cond_list = [0, 1, 2, 0, 1, 2, 0, 1, 2]
ges_cond_list = [0, 0, 0, 1, 1, 1, 2, 2, 2]
lexical_cond = [0, 1, 1, 2, 3, 3, 2, 3, 3]
all_cond_data = np.zeros([19, 9])
lexical_cond_data = np.zeros([19, 9])
ges_lex = [0, 0, 0, 1, 1, 1, 1, 1, 1]
spe_lex = [0, 1, 1, 0, 1, 1, 0, 1, 1]
ges_lex_cond_data = np.zeros([19, 9])
spe_lex_cond_data = np.zeros([19, 9])

for col in np.arange(9):
    spe_cond_data[:, col] = [spe_cond_list[col] for row in np.arange(19)]
    ges_cond_data[:, col] = [ges_cond_list[col] for row in np.arange(19)]
    all_cond_data[:, col] = [col for row in np.arange(19)]
    lexical_cond_data[:, col] = [lexical_cond[col] for row in np.arange(19)]
    ges_lex_cond_data[:, col] = [ges_lex[col] for row in np.arange(19)]
    spe_lex_cond_data[:, col] = [spe_lex[col] for row in np.arange(19)]

flat_ges_cond = ges_cond_data.flatten()
flat_spe_cond = spe_cond_data.flatten()
flat_cond = all_cond_data.flatten()
flat_lexical = lexical_cond_data.flatten()
flat_ges_lex = ges_lex_cond_data.flatten()
flat_spe_lex = spe_lex_cond_data.flatten()

flat_mi = np.array(mi_data).flatten()
flat_ig_ges = np.array(ig_ges_data).flatten()
flat_ig_spe = np.array(ig_spe_data).flatten()
flat_ig_unit = np.array(entropy_loss).flatten()
flat_rt = np.array(rt_data.T).flatten()

component_list = [(0, 100), (300, 500), (600, 800)]  # time_window

# 7 electrode ROI
component_chan = {'LA': ['F1', 'F3', 'F5', 'FC1', 'FC3', 'FC5'],
                  'LC': ['C1', 'C3', 'C5', 'CP1', 'CP3', 'CP5'],
                  'LP': ['P1', 'P3', 'P5', 'PO3', 'PO5', 'O1'],
                  'RA': ['F2', 'F4', 'F6', 'FC2', 'FC4', 'FC6'],
                  'RC': ['C2', 'C4', 'C6', 'CP2', 'CP4', 'CP6'],
                  'RP': ['P2', 'P4', 'P6', 'PO4', 'PO6', 'O2'],
                  'ML': ['FZ', 'FCZ', 'CZ', 'PZ', 'OZ', 'CPZ']}

condition_list = np.arange(9)
eeg_data = np.zeros([2, len(component_list), len(component_chan), len(stimulus_list), len(condition_list)])
for comp_index, component in enumerate(component_list):
    time_index = [int(component[k] / 2 + 100) for k in np.arange(2)]
    time_num_index = np.arange(time_index[0], time_index[1], 1)
    for chan_index, chan in enumerate(component_chan):
        cond_index = 0
        for dp_index, dpn in enumerate(dp_cond):
            for ip_index, ipn in enumerate(ip_cond):
                keep_data = np.empty([0, len(component_chan[chan]), time_point])
                average_keep_data = np.empty([0, len(component_chan[chan]), time_point])
                stim_label = np.empty([0])
                average_stim_label = np.empty([0])
                for sub_index, sub in enumerate(sub_list):
                    diff_data = np.zeros([2, len(stimulus_list), len(component_chan[chan]),
                                          time_point])
                    for con_index, congru in enumerate(congru_cond):
                        cond_folder = work_directory + '/' + str(dpn) + '_' + str(ipn) + '_' + str(congru) + '/'
                        event_mat = sio.loadmat(cond_folder + str(sub) + '.mat')

                        delete_id, new_event_df = find_useless_trial(event_mat, stim_info)

                        sub_epochs = mne.read_epochs_eeglab(cond_folder + str(sub) + '.set')
                        sub_epochs.pick_channels(ch_names=component_chan[chan])

                        for stim_index, stim_name in enumerate(stimulus_list):

                            keep_trial = (new_event_df.loc[:, 'dp_stim'] == stim_index)

                            if np.sum(keep_trial) == 0:
                                diff_data[con_index, stim_index] = np.nan
                            else:
                                stim_epoch = sub_epochs[keep_trial].copy()
                                diff_data[con_index, stim_index] = stim_epoch.average()._data

                    different = diff_data[1, :] - diff_data[0, :]
                    stim_bool = np.array([~np.isnan(d) for d in different[:, 0, 0]])
                    different = different[stim_bool]
                    effect_stim = np.arange(19)[stim_bool]
                    stim_label = np.concatenate((stim_label, effect_stim))
                    keep_data = np.concatenate((keep_data, different))

                    average = diff_data.mean(axis=0)
                    average_stim_bool = np.array([~np.isnan(d) for d in average[:, 0, 0]])
                    average = average[average_stim_bool]
                    average_stim = np.arange(19)[average_stim_bool]
                    average_stim_label = np.concatenate((average_stim_label, average_stim))

                    average_keep_data = np.concatenate((average_keep_data, average))

                stim_data = np.zeros([19, len(component_chan[chan]), time_point])
                average_data = np.zeros([19, len(component_chan[chan]), time_point])

                for stim_index, stim_name in enumerate(stimulus_list):
                    stim_diff = keep_data[stim_label == stim_index]
                    stim_data[stim_index] = stim_diff.mean(axis=0)

                    stim_average = average_keep_data[average_stim_label == stim_index]
                    average_data[stim_index] = stim_average.mean(axis=0)

                target_data = stim_data[:, :, time_num_index].mean(axis=(1, 2))
                target_average_data = average_data[:, :, time_num_index].mean(axis=(1, 2))
                eeg_data[0, comp_index, chan_index, :, cond_index] = target_data  # saving different wave amplitude
                eeg_data[1, comp_index, chan_index, :, cond_index] = target_average_data  # saving ERP amplitude
                cond_index += 1

np.save(oj(work_directory, 'all_eeg_average_data.npy'), eeg_data)

gesture_entropy_x = pd.concat((g_entropy.loc[:, 0.75], g_entropy.loc[:, 0.75], g_entropy.loc[:, 0.75],
                               g_entropy.loc[:, 1.0], g_entropy.loc[:, 1.0], g_entropy.loc[:, 1.0],
                               g_entropy.loc[:, 1.25], g_entropy.loc[:, 1.25], g_entropy.loc[:, 1.25]), axis=1)
gesture_entropy_x.columns = pd.Index(all_cond_name)
speech_entropy_x = pd.concat((s_entropy, s_entropy, s_entropy), axis=1)
speech_entropy_x.columns = pd.Index(all_cond_name)

information_weight_correlation = np.zeros([6, 2, len(component_list), len(component_chan)])
information_weight_p = np.zeros([6, 2, len(component_list), len(component_chan)])

information_indicator = np.array([flat_ig_unit, flat_ig_ges, flat_ig_spe, flat_mi,
                                  np.array(gesture_entropy_x).flatten(), np.array(speech_entropy_x).flatten()])

# Gain(unit) Gain(ges) Gain(spe) mi ges_entropy spe_entropy

method = 'spearman'
for information_type in np.arange(6):
    single_information = information_indicator[information_type]
    for eeg_type in np.arange(2):
        for comp_index, component in enumerate(component_list):
            for chan_index, chan in enumerate(component_chan):
                single_eeg = eeg_data[eeg_type, comp_index, chan_index].flatten()
                corr_stats = pg.corr(single_information, single_eeg, method=method)
                information_weight_correlation[information_type, eeg_type, comp_index, chan_index] = corr_stats.loc[
                    method, 'r']
                information_weight_p[information_type, eeg_type, comp_index, chan_index] = corr_stats.loc[
                    method, 'p-val']

np.save(oj(work_directory, 'information_weight_p.npy'), information_weight_p)
np.save(oj(work_directory, 'information_weight_correlation.npy'), information_weight_correlation)

# -*- coding = utf-8 -*-
# @Time : 30/09/2024 21.38
# @Author : yangjw
# @Site : 
# @File : Group_contrast_for_neural_feature.py
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
from statsmodels.stats.multitest import fdrcorrection


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

mode_epochs = mne.read_epochs_eeglab(work_directory + '/1_1_1/1.set')

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

# generating gesture dp erp
all_average = np.zeros([len(dp_cond), len(sub_list), chan_num, time_point])
for dp_index, dpn in enumerate(dp_cond):
    for sub_index, sub in enumerate(sub_list):
        cond_erp = np.zeros([3, 2, chan_num, time_point])
        for ip_index, ipn in enumerate(ip_cond):
            for con_index, congru in enumerate(congru_cond):

                cond_folder = work_directory + '/' + str(dpn) + '_' + str(ipn) + '_' + str(congru) + '/'
                event_mat = sio.loadmat(cond_folder + str(sub) + '.mat')

                delete_id, new_event_df = find_useless_trial(event_mat, stim_info)

                sub_epochs = mne.read_epochs_eeglab(cond_folder + str(sub) + '.set')

                single_erp = sub_epochs._data.mean(axis=0)

                cond_erp[ip_index, con_index] = single_erp

        sub_mean_erp = cond_erp.mean(axis=(0, 1))
        all_average[dp_index, sub_index] = sub_mean_erp

np.save(oj(work_directory, 'all_dp_average_data.npy'), all_average)
all_average = np.load(oj(work_directory, 'all_dp_average_data.npy'))

# generating speech ip erp
all_ip_average = np.zeros([len(ip_cond), len(sub_list), chan_num, time_point])

for ip_index, ipn in enumerate(ip_cond):
    for sub_index, sub in enumerate(sub_list):
        cond_erp = np.zeros([3, 2, chan_num, time_point])
        for dp_index, dpn in enumerate(dp_cond):
            for con_index, congru in enumerate(congru_cond):
                cond_folder = work_directory + '/' + str(dpn) + '_' + str(ipn) + '_' + str(congru) + '/'
                event_mat = sio.loadmat(cond_folder + str(sub) + '.mat')

                delete_id, new_event_df = find_useless_trial(event_mat, stim_info)

                sub_epochs = mne.read_epochs_eeglab(cond_folder + str(sub) + '.set')
                # sub_epochs.pick_channels(ch_names=['CZ'])

                epoch_data = sub_epochs._data.copy()

                scaler = StandardScaler()
                z_scores = scaler.fit_transform(epoch_data.flatten().reshape(-1, 1))
                trans_data = z_scores.reshape(epoch_data.shape)

                single_erp = trans_data.mean(axis=0)

                cond_erp[dp_index, con_index] = single_erp

        sub_mean_erp = cond_erp.mean(axis=(0, 1))
        all_ip_average[ip_index, sub_index] = sub_mean_erp

np.save(oj(work_directory, 'all_ip_average_data.npy'), all_ip_average)
all_ip_average = np.load(oj(work_directory, 'all_ip_average_data.npy'))


# generating congruency erp
all_congruency_average = np.zeros([len(congru_cond), len(sub_list), chan_num, time_point])

for con_index, congru in enumerate(congru_cond):
    for sub_index, sub in enumerate(sub_list):
        cond_erp = np.zeros([3, 3, chan_num, time_point])
        for ip_index, ipn in enumerate(ip_cond):
            for dp_index, dpn in enumerate(dp_cond):
                cond_folder = work_directory + '/' + str(dpn) + '_' + str(ipn) + '_' + str(congru) + '/'
                event_mat = sio.loadmat(cond_folder + str(sub) + '.mat')

                delete_id, new_event_df = find_useless_trial(event_mat, stim_info)

                sub_epochs = mne.read_epochs_eeglab(cond_folder + str(sub) + '.set')
                # sub_epochs.pick_channels(ch_names=['CZ'])
                epoch_data = sub_epochs._data.copy()

                scaler = StandardScaler()
                z_scores = scaler.fit_transform(epoch_data.flatten().reshape(-1, 1))
                trans_data = z_scores.reshape(epoch_data.shape)

                single_erp = trans_data.mean(axis=0)

                cond_erp[ip_index, dp_index] = single_erp

        sub_mean_erp = cond_erp.mean(axis=(0, 1))
        all_congruency_average[con_index, sub_index] = sub_mean_erp

np.save(oj(work_directory, 'all_congruency_average_data.npy'), all_congruency_average)
all_congruency_average = np.load(oj(work_directory, 'all_congruency_average_data.npy'))

component_list = [(0, 100), (300, 500), (600, 800)]
component_chan = {'LA': ['F1', 'F3', 'F5', 'FC1', 'FC3', 'FC5'],
                  'LC': ['C1', 'C3', 'C5', 'CP1', 'CP3', 'CP5'],
                  'LP': ['P1', 'P3', 'P5', 'PO3', 'PO5', 'O1'],
                  'RA': ['F2', 'F4', 'F6', 'FC2', 'FC4', 'FC6'],
                  'RC': ['C2', 'C4', 'C6', 'CP2', 'CP4', 'CP6'],
                  'RP': ['P2', 'P4', 'P6', 'PO4', 'PO6', 'O2'],
                  'ML': ['FZ', 'FCZ', 'CZ', 'PZ', 'OZ', 'CPZ']}

ges_cond_str = ['BD', 'DP', 'AD']
spe_cond_str = ['BI', 'IP', 'AI']
ges_F = np.zeros([len(component_chan), len(component_list)])
ges_p = np.zeros([len(component_chan), len(component_list)])
ges_n2 = np.zeros([len(component_chan), len(component_list)])
spe_F = np.zeros([len(component_chan), len(component_list)])
spe_p = np.zeros([len(component_chan), len(component_list)])
spe_n2 = np.zeros([len(component_chan), len(component_list)])
congru_t = np.zeros([len(component_chan), len(component_list)])
congru_p = np.zeros([len(component_chan), len(component_list)])
congru_cohen = np.zeros([len(component_chan), len(component_list)])
for chan_index, chan in enumerate(component_chan):
    chan_list = mode_epochs.ch_names
    real_chan_index = np.array([chan_list.index(c) for c in component_chan[chan]])
    for time_i, time_cls in enumerate(component_list):
        time_index = [int(time_cls[k] / 2 + 100) for k in np.arange(2)]
        time_num_index = np.arange(time_index[0], time_index[1], 1)
        gesture_eeg_feature = all_average[:, :, real_chan_index][:, :, :, time_num_index].mean(axis=(2, 3))
        ges_flat = gesture_eeg_feature.flatten()
        ges_cond = [ges_cond_str[dpi] for dpi in np.arange(3) for sub_index in sub_list]
        ges_sub = [sub_index for dpi in np.arange(3) for sub_index in sub_list]
        gesture_df = pd.DataFrame({
            'eeg_data': ges_flat,
            'gesture_DP': ges_cond,
            'subject': ges_sub
        })
        ges_anova = pg.rm_anova(data=gesture_df, dv='eeg_data', within='gesture_DP', subject='subject')
        ges_F[chan_index, time_i] = ges_anova.loc[0, 'F']
        ges_p[chan_index, time_i] = ges_anova.loc[0, 'p-unc']
        ges_n2[chan_index, time_i] = ges_anova.loc[0, 'ng2']

        speech_eeg_feature = all_ip_average[:, :, real_chan_index][:, :, :, time_num_index].mean(axis=(2, 3))
        spe_flat = speech_eeg_feature.flatten()
        spe_cond = [spe_cond_str[ipi] for ipi in np.arange(3) for sub_index in sub_list]
        spe_sub = [sub_index for ipi in np.arange(3) for sub_index in sub_list]
        speech_df = pd.DataFrame({
            'eeg_data': spe_flat,
            'speech_IP': spe_cond,
            'subject': spe_sub
        })
        spe_anova = pg.rm_anova(data=speech_df, dv='eeg_data', within='speech_IP', subject='subject')
        spe_F[chan_index, time_i] = spe_anova.loc[0, 'F']
        spe_p[chan_index, time_i] = spe_anova.loc[0, 'p-unc']
        spe_n2[chan_index, time_i] = spe_anova.loc[0, 'ng2']

        congru_eeg_feature = all_congruency_average[:, :, real_chan_index][:, :, :, time_num_index].mean(axis=(2, 3))
        congru_t_test = pg.ttest(congru_eeg_feature[1], congru_eeg_feature[0], paired=True)
        congru_t[chan_index, time_i] = congru_t_test.loc['T-test', 'T']
        congru_p[chan_index, time_i] = congru_t_test.loc['T-test', 'p-val']
        congru_cohen[chan_index, time_i] = congru_t_test.loc['T-test', 'cohen-d']


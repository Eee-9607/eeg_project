# -*- coding = utf-8 -*-
# @Time : 30/09/2024 20.45
# @Author : yangjw
# @Site : 
# @File : Full_factorial_RSA.py
# @Software : PyCharm
# @contact: jwyang9826@gmail.com


import mne
import numpy as np
import pandas as pd
import scipy.io as sio
import os
import rsatoolbox
from jw_function.supply_function import create_dir
from jw_function.bhv_function import find_useless_trial
from jw_function.rsa_function import upper_tri
import pingouin as pg
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from neurora.stuff import clusterbased_permutation_1d_1samp_1sided
from skimage.measure import label
from os.path import join as oj

work_directory = os.getcwd()
project_path = work_directory
fig_path = project_path + '/figure/'
behavior_path = oj(work_directory, 'behavioral_information')

excel_path = oj(behavior_path, 'stimuli_information_update.xls')
stim_info = pd.read_excel(excel_path)

mi_path = project_path + '/entropy/mutual_information_full_item.xlsx'
mi_entropy = pd.read_excel(mi_path)

create_dir(fig_path)

all_sub = np.arange(1, 40)
miss_sub = [7, 8, 10, 17, 20, 21, 23, 24, 25]
sub_list = [sub for sub in all_sub if sub not in miss_sub]

dp_cond = np.arange(1, 4)
ip_cond = np.arange(1, 4)
congru_cond = np.arange(1, 3)

chan_num = 42

time_point = 600

# drop_channel = np.load(project_path + '/drop_channel.npy')
drop_channel = []
time_win = 10
# construct 3 different theory RDMs
all_sub_rdm = np.zeros([3, len(sub_list), int((18 * 18 - 18) / 2)])
all_eeg_rdm = np.zeros([time_point, len(sub_list), int((18 * 18 - 18) / 2)])

for sub_index, sub in enumerate(sub_list):

    effect_list = np.zeros([18])
    dp_list = np.zeros([18])
    ip_list = np.zeros([18])

    cond_index = 0

    condition_array = []
    data_array = np.empty(shape=[0, chan_num - len(drop_channel), time_point])
    for dp_index, d_cond in enumerate(dp_cond):
        for ip_index, i_cond in enumerate(ip_cond):
            for con_index, congru in enumerate(congru_cond):
                dp_list[cond_index] = dp_index
                ip_list[cond_index] = ip_index
                effect_list[cond_index] = con_index
                cond_folder = project_path + '/' + str(d_cond) + '_' + str(i_cond) + '_' + str(congru) + '/'
                event_mat = sio.loadmat(cond_folder + str(sub) + '.mat')

                delete_id, new_event_df = find_useless_trial(event_mat, stim_info)

                sub_epochs = mne.read_epochs_eeglab(cond_folder + str(sub) + '.set')
                # sub_epochs.drop_channels(drop_channel)
                sub_epochs.drop(delete_id)
                sub_epochs.apply_baseline(baseline=(-0.2, 0))

                data_array = np.concatenate((data_array, sub_epochs._data.copy()), axis=0)

                con_array = [cond_index for t in np.arange(len(sub_epochs))]
                condition_array.extend(con_array)
                cond_index += 1

    # transform RDM into upper_triangle RDM
    congru_RDM = np.zeros([18, 18])
    dp_RDM = np.zeros([18, 18])
    ip_RDM = np.zeros([18, 18])
    for row in np.arange(18):
        for col in np.arange(18):
            congru_RDM[row, col] = abs(effect_list[row] - effect_list[col])
            dp_RDM[row, col] = abs(dp_list[row] - dp_list[col])
            ip_RDM[row, col] = abs(ip_list[row] - ip_list[col])

    all_sub_rdm[0, sub_index] = upper_tri(congru_RDM)
    all_sub_rdm[1, sub_index] = upper_tri(dp_RDM)
    all_sub_rdm[2, sub_index] = upper_tri(ip_RDM)

    # fit the eeg data into rsatoolbox object
    data = rsatoolbox.data.TemporalDataset(
        data_array,
        channel_descriptors={'names': sub_epochs.ch_names},
        obs_descriptors={'conds': condition_array},
        time_descriptors={'time': sub_epochs.times},
    )
    data.sort_by('conds')

    # calculate eeg temporal rdm
    rdms_data = rsatoolbox.rdm.calc_rdm_movie(data, method='euclidean',
                                              descriptor='conds')
    all_eeg_rdm[:, sub_index, :] = rdms_data.dissimilarities.copy()

all_beta = np.zeros([3, time_point, len(sub_list)])
all_corr = np.zeros([3, time_point, len(sub_list)])
t_array = np.zeros([3, time_point, len(sub_list)])  # perform t test of beta diff: 0 - 1, 0 - 2, 1 - 2
index_list = [(0, 1), (0, 2), (1, 2)]  # mark the index
for tp in np.arange(time_point):
    for sub_index, sub in enumerate(sub_list):
        squeeze_array = np.zeros([3, int((18 * 18 - 18) / 2)])
        for rdm_index in np.arange(3):
            all_corr[rdm_index, tp, sub_index] = rsatoolbox.rdm.compare(all_sub_rdm[rdm_index, sub_index],
                                                                        all_eeg_rdm[tp, sub_index], method='spearman')

            first, second = index_list[rdm_index]

            X = np.concatenate((all_sub_rdm[:, sub_index].T, np.ones([int((18 * 18 - 18) / 2), 1])), axis=1)
            Y = all_eeg_rdm[tp, sub_index, :]
            m = 3
            n = len(all_eeg_rdm[tp, sub_index, :])
            beta = (np.linalg.inv(X.T @ X)) @ (X.T @ Y)
            rss = Y.T @ Y - beta.T @ X.T @ Y
            C = np.diag(np.linalg.inv(X.T @ X))
            sigma = np.sqrt(rss / (n - m))

            cov_A = (sigma ** 2) * np.linalg.inv(X.T @ X)  # calculating covariance matrix
            cov = cov_A[first, second]
            vp = np.sqrt((np.sqrt(C[first]) * sigma) ** 2 + (np.sqrt(C[second]) * sigma) ** 2 - 2 * cov)

            # stats = pg.linear_regression(all_rdm, SL_RDM.dissimilarities[voxel_index, :])
            t_array[rdm_index, tp, sub_index] = (beta[first] - beta[second]) / vp

        beta_stats = pg.linear_regression(all_sub_rdm[:, sub_index].T, all_eeg_rdm[tp, sub_index])
        all_beta[0, tp, sub_index] = beta_stats.loc[1, 'T']
        all_beta[1, tp, sub_index] = beta_stats.loc[2, 'T']
        all_beta[2, tp, sub_index] = beta_stats.loc[3, 'T']

# extract significant time window
t_judge = np.zeros([3, time_point])
beta_judge = np.zeros([3, time_point])

for cp_index in np.arange(3):
    for tp in np.arange(time_point):
        t_group_stats = stats.ttest_1samp(t_array[cp_index, tp], popmean=0)
        beta_group_stats = stats.ttest_1samp(all_beta[cp_index, tp], popmean=0)

        if t_group_stats.pvalue < 0.05:
            t_judge[cp_index, tp] = 1

        if beta_group_stats.pvalue < 0.05:
            beta_judge[cp_index, tp] = 1

rdm_color = ['#CC011F', '#8CC5BE', '#6A8EC9']
# rdm_color = ['palegreen', 'wheat', 'lightskyblue']
rdm_label = ['Congruency', 'Gesture', 'Speech']

# parameter setting
p_value = 0.01
alpha_list = [0.8, 0.4, 0.8]
zorder_list = [2, 0, 1]
min_list = [min(all_beta[r_index, :].mean(axis=1)) for r_index in np.arange(3)]
max_list = [max(all_beta[r_index, :].mean(axis=1)) for r_index in np.arange(3)]
fig, ax = plt.subplots(1, 1)

# split first gesture cluster into 2 sub-cluster to identify ERP component
cls_time_list = [(0, 100), (150, 250)]
cls_time_index = [(int((m[0] + 200) / 2), int((m[1] + 200) / 2)) for m in cls_time_list]

for cls_index, cls_t in enumerate(cls_time_list):
    cls_t_index = np.arange(cls_t[0], cls_t[1] + 1, 2)
    cls_i = np.arange(cls_time_index[cls_index][0], cls_time_index[cls_index][1] + 1, 1)
    plot_min_y = [min(min_list) for c in cls_t_index]
    plot_max_y = all_beta[1, cls_i].mean(axis=1)
    if cls_index == 0:
        ax.fill_between(cls_t_index, plot_min_y, plot_max_y, color=rdm_color[1], alpha=0.8)

# plot significant beta time cluster
all_rdm_stats = np.empty(shape=[0, time_point])
for rdm_index, rdm_cl in enumerate(rdm_color):
    stats_results = clusterbased_permutation_1d_1samp_1sided(all_beta[rdm_index].transpose([1, 0]),
                                                             level=0,
                                                             p_threshold=p_value,
                                                             clusterp_threshold=p_value,
                                                             iter=1000)
    all_rdm_stats = np.concatenate((all_rdm_stats, stats_results.reshape([1, len(stats_results)])), axis=0)


    ax.plot(sub_epochs.times * 1000, all_beta[rdm_index, :].mean(axis=1), color=rdm_cl,
            label=rdm_label[rdm_index], alpha=1, zorder=zorder_list[rdm_index])
    labels = label(stats_results, connectivity=1)
    nclusters = int(np.max(labels))

    for k in np.arange(1, nclusters + 1):
        sig_time = (labels == k)

        hline = np.array([min(min_list) for i in np.arange(np.sum(sig_time))])

        # setting opacity mannually
        if ((k == 2) and (rdm_index == 0)):
            alpha = 0.3
        elif ((k == 1) and (rdm_index == 2)):
            alpha = 0.4
        elif ((k == 2) and (rdm_index == 2)):
            alpha = 0.4
        else:
            alpha = alpha_list[rdm_index]

        ax.fill_between(sub_epochs.times[sig_time] * 1000, hline,
                        all_beta[rdm_index, :].mean(axis=1)[sig_time],
                        facecolor=rdm_cl, zorder=zorder_list[rdm_index], alpha=alpha)

    ax.set_xlabel('times(ms)', fontdict={'fontsize': 20, 'weight': 'bold'}, labelpad=10)
    ax.set_ylabel('Beta', fontdict={'fontsize': 20, 'weight': 'bold'}, labelpad=10)


ax.set_ylim([min(min_list), max(max_list)])
ax.tick_params(left=False, labelsize=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
fig.subplots_adjust(bottom=0.2, top=0.9)
fig.legend(frameon=False, prop={'weight': 'bold', 'size': 18})
plt.show()

np.save(project_path + '/full_factorial_RSA_all_beta.npy', all_beta)
np.save(project_path + '/full_factorial_RSA_onw-sided_results.npy', all_rdm_stats)



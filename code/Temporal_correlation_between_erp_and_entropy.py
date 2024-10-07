# -*- coding = utf-8 -*-
# @Time : 07/10/2024 10.11
# @Author : yangjw
# @Site : 
# @File : Temporal_correlation_between_erp_and_entropy.py
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


work_directory = os.getcwd()
fig_path = oj(work_directory, 'figure')

avg_path = oj(work_directory, 'data', 'avg_data')

excel_path = oj(work_directory, 'stimuli_information_update.xls')
stim_info = pd.read_excel(excel_path)

entropy_path = oj(work_directory, 'entropy')

g_entropy_path = oj(entropy_path, 'gesture_entropy_full_item.xlsx')
g_entropy = pd.read_excel(g_entropy_path, index_col=0)
s_entropy_path = oj(entropy_path, 'speech_entropy_full_item.xlsx')
s_entropy = pd.read_excel(s_entropy_path, index_col=0)
item_rank = pd.read_excel(oj(entropy_path, 'itemrank2.xls'), header=None)

dp_cond = np.arange(1, 4)
ip_cond = np.arange(1, 4)

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

time_series = np.arange(-199, 1001, 2)
method = 'spearman'

cond_time_dict = {'Entropy(gesture)': [(0, 100), (0, 100), (0, 100),],
                  'Entropy(speech)': [(638, 686), (648, 668), (776, 780)]}

cond_chan_dict = {'Entropy(gesture)': [(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19),
                                       (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19),
                                       (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19)],
                  'Entropy(speech)': [(3, 4, 5, 11, 12), (3, 4, 5, 11), (4, 11, 18, 5, 12, 10, 3, 20, 9, 17, 19, 13)],
                  }

mode_epochs = mne.read_epochs_eeglab(work_directory + '/1_1_1/1.set')

dp_color_list = ['#BDE9C1', '#98CBC4', '#4DC385']
ip_color_list = ['#A5CDE5', '#88A4D3', '#3535C1']
target_metric = 'Entropy(gesture)'

# plot bar figure

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

line_list = []
for cond_index, cond in enumerate(dp_cond):
    sig_time = cond_time_dict[target_metric][cond_index]
    ax.plot((0, 1000), (2 - cond_index, 2 - cond_index), color='#e8e8e8', linewidth=20, zorder=0)
    ax.plot(sig_time, (2 - cond_index, 2 - cond_index), color=ip_color_list[cond_index], linewidth=20, zorder=1)

ax.set_yticks([])
ax.set_ylabel(ylabel='')
ax.set_yticklabels([])
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([])
ax.set_xlabel(xlabel='')
ax.set_xticklabels([])

fig.tight_layout()
fig.subplots_adjust(left=0.1, right=0.98, wspace=0.3, bottom=0.13)
fig.savefig(oj(fig_path, 'ip_bar_figure.png'), dpi=600)
plt.show()


target_metric = 'Entropy(speech)'
method = 'spearman'
target_entropy = s_entropy
index_list = [0.75, 1.0, 1.25]
ges_loading_name = ['before', 'ip', 'after']
target_color = ip_color_list
all_correlation = np.zeros([len(dp_cond)])
all_p = np.zeros([len(dp_cond)])
fig, axes = plt.subplots(len(dp_cond), 1, figsize=(4, 8))
# now we should plot dp cluster and corresponding erp

for dp_index, dp in enumerate(dp_cond):
    ax = axes[dp_index]
    roi_channel = cond_chan_dict[target_metric][dp_index]
    roi_channel = [chan - 1 for chan in roi_channel]
    roi_time = cond_time_dict[target_metric][dp_index]
    time_index = [int(roi_time[k] / 2 + 100) for k in np.arange(2)]
    time_num_index = np.arange(time_index[0], time_index[1] + 1, 1)

    cond_file = os.path.join(avg_path, f'EEG_avg_{ges_loading_name[dp_index]}.mat')
    cond_mat = sio.loadmat(cond_file)
    cond_data = cond_mat[f'EEG_avg_{ges_loading_name[dp_index]}']

    plot_data = cond_data[:, :, roi_channel, :][:, :, :, time_num_index].mean(axis=(1, 2, 3))

    corr = pg.corr([round(d, 2) for d in plot_data], round(target_entropy.loc[item_rank.loc[:, 0], index_list[dp_index]], 2), method=method)
    all_correlation[dp_index] = corr.loc[method, 'r']
    all_p[dp_index] = corr.loc[method, 'p-val']

    plot_df = pd.DataFrame({'eeg': plot_data,
                            'it': target_entropy.loc[item_rank.loc[:, 0], index_list[dp_index]]})
    sns.regplot(plot_df, x='it', y='eeg', ax=ax, color=target_color[dp_index])

    ax.set_xlabel(xlabel=target_metric, fontdict={'fontsize': 17, 'weight': 'bold'})
    # ax.set_xticklabels(fontdict={'fontsize': 14, 'weight': 'bold'})
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylabel(ylabel='Amplitude(uv)', fontdict={'fontsize': 17, 'weight': 'bold'})

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for label in ax.get_xticklabels():
        label.set_fontweight('bold')

    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

fig.tight_layout()
fig.subplots_adjust(left=0.25, right=0.95, wspace=0.3, bottom=0.13)
fig.savefig(oj(fig_path, 'speech_entropy_regression_figure.png'), dpi=600)
plt.show()

# plot entropy topography

target_metric = 'Entropy(gesture)'
target_color = dp_color_list
channel_map = matplotlib.colormaps['RdBu_r']
fig, axes = plt.subplots(len(dp_cond), figsize=(12, 8))
cond_index = 0
for dp_index, dp in enumerate(dp_cond):
    ax = axes[dp_index]
    roi_channel = np.array(cond_chan_dict[target_metric][cond_index]) - 1

    plot_bool = [i in roi_channel for i in np.arange(len(mode_epochs.ch_names))]

    true_list = np.array([True for i in np.arange(len(mode_epochs.ch_names))])
    mask_para = dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=0)

    im = mne.viz.plot_topomap(np.zeros([len(mode_epochs.ch_names)]), mode_epochs.info, res=600, cmap=channel_map,
                              names=None, axes=ax, show=False, vlim=[-1, 1], mask=true_list, mask_params=mask_para)

    all_position = np.zeros([len(mode_epochs.ch_names), 3])
    for chan in np.arange(len(mode_epochs.ch_names)):
        all_position[chan] = mode_epochs.info['dig'][chan + 3]['r']

    x, y = all_position[plot_bool, :2].T * 0.8
    if roi_channel[0] + 1 == 0:
        ax.scatter(x, y, s=80, c='grey', marker='o', edgecolor='k', linewidth=0)
    else:
        ax.scatter(x, y, s=80, c=target_color[cond_index], marker='o', edgecolor='k', linewidth=0)

    cond_index += 1

fig.savefig(oj(fig_path, 'gesture_entropy_topography.png'), dpi=600)
plt.show()
# -*- coding = utf-8 -*-
# @Time : 07/10/2024 09.28
# @Author : yangjw
# @Site : 
# @File : Temporal_correlation_between_different_wave_and_information_gain.py
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


work_directory = os.getcwd()
fig_path = oj(work_directory, 'figure')
behavior_path = oj(work_directory, 'behavioral_information')

excel_path = oj(behavior_path, 'stimuli_information_update.xls')
stim_info = pd.read_excel(excel_path)

entropy_path = oj(work_directory, 'entropy')

entropy_loss = pd.read_excel(oj(entropy_path, "entropy_loss.xlsx"), index_col=0)

item_rank = pd.read_excel(oj(entropy_path, 'itemrank2.xls'), header=None)

# reading four information gains and corresponding different wave amplitude
ges_gain = pd.read_spss(oj(entropy_path, 'correlation_ges_gain.sav'))
spe_gain = pd.read_spss(oj(entropy_path, 'correlation_spe_gain.sav'))
mi = pd.read_spss(oj(entropy_path, 'correlation_mi.sav'))
unit_gain = pd.read_spss(oj(entropy_path, 'correlation_unit_gain.sav'))

dp_condition = ['bd', 'dp', 'ad']
ip_condition = ['bi', 'ip', 'ai']
mi_column = pd.Index([f"{dp}_{ip}_{n}" for dp in dp_condition for ip in ip_condition for n in ['mi', 'eeg']])
mi.columns = mi_column
unit_column = pd.Index([f"{dp}_{ip}_{n}" for dp in dp_condition for ip in ip_condition for n in ['unit_gain', 'eeg']])
unit_gain.columns = unit_column

correct_idx = [i - k for i in np.arange(1, spe_gain.shape[1], 2) for k in np.arange(2)]
spe_gain = spe_gain.iloc[:, correct_idx]
spe_column = pd.Index([f"{dp}_{ip}_{n}" for dp in dp_condition for ip in ip_condition for n in ['spe_gain', 'eeg']])
spe_gain.columns = spe_column

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

metric_name_list = ['ges_gain', 'spe_gain', 'unit_gain', 'mi']
metric_list = [ges_gain, spe_gain, unit_gain, mi]
plot_name_list = ['Gain(gesture)', 'Gain(speech)', 'Gain(unit)', 'MI']

ges_bool = [1, 0, 0, 0, 1, 0, 0, 0, 1]
spe_bool = [0, 1, 1, 0, 1, 1, 0, 1, 1]
unit_bool = [0, 1, 1, 1, 1, 1, 1, 1, 1]
mi_bool = [1, 0, 0, 0, 1, 0, 0, 0, 1]

bool_list = [ges_bool, spe_bool, unit_bool, mi_bool]

plot_cond_color_list = ['salmon', 'violet', 'lightskyblue',
                        'orangered', 'darkviolet', 'deepskyblue',
                        'brown', 'indigo', 'midnightblue']

method = 'spearman'

all_correlation = np.zeros([len(metric_name_list), len(dp_cond), len(ip_cond)])
all_p = np.zeros([len(metric_name_list), len(dp_cond), len(ip_cond)])
# next we plot correlation figure in 9 conditions

for name_index, name in enumerate(metric_name_list):
    fig, axes = plt.subplots(len(dp_cond), len(ip_cond), figsize=(12, 8))
    cond_index = 0
    for dp_index, dp in enumerate(dp_cond):
        for ip_index, ip in enumerate(ip_cond):

            ax = axes[dp_index, ip_index]

            plot_df = pd.DataFrame({'eeg': metric_list[name_index].iloc[:, 1 + cond_index * 2],
                                    'it': metric_list[name_index].iloc[:, cond_index * 2]})

            if bool_list[name_index][cond_index]:
                sns.regplot(plot_df, x='it', y='eeg', ax=ax, color=plot_cond_color_list[cond_index])
            else:
                sns.regplot(plot_df, x='it', y='eeg', ax=ax, color='grey')
            corr_stats = pg.corr(plot_df['eeg'],
                                 plot_df['it'],
                                 method=method)
            all_correlation[name_index, dp_index, ip_index] = corr_stats.loc[method, 'r']
            all_p[name_index, dp_index, ip_index] = corr_stats.loc[method, 'p-val']


            ax.set_xlabel(xlabel=plot_name_list[name_index], fontdict={'fontsize': 14, 'weight': 'bold'})
            ticks = ax.get_xticks()
            new_ticks = [t for t in ticks if t in np.arange(0, 10, 0.5)]
            labels = [f"{x:.1f}" for x in new_ticks]
            ax.set_xticks(new_ticks)
            ax.set_xticklabels(labels, fontdict={'fontsize': 14, 'weight': 'bold'})

            # ax.set_yticks([-4, -2, 0, 2, 4])
            # ax.set_ylabel(ylabel='Semantic congruency', fontdict={'fontsize': 14, 'weight': 'bold'})
            # ax.set_yticklabels([-4, -2, 0, 2, 4], fontdict={'fontsize': 14, 'weight': 'bold'})

            for label in ax.get_yticklabels():
                label.set_fontweight('bold')
                label.set_fontsize(14)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            cond_index += 1

    fig.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.98, wspace=0.3, bottom=0.13)
    fig.savefig(oj(fig_path, f'{metric_name_list[name_index]}_regression_figure.png'), dpi=600)
    plt.show()

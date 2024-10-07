# -*- coding = utf-8 -*-
# @Time : 07/10/2024 09.01
# @Author : yangjw
# @Site : 
# @File : Mediation_model.py
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
from semopy import Model
from semopy.examples import holzinger39
import semopy
from pyprocessmacro import Process
from pyprocessmacro.models import bootstrap_sampler, bias_corrected_ci

work_directory = os.getcwd()
fig_path = oj(work_directory, 'figure')

behavior_path = oj(work_directory, 'behavioral_information')

excel_path = oj(behavior_path, 'stimuli_information_update.xls')
stim_info = pd.read_excel(excel_path)

entropy_path = oj(work_directory, 'entropy')

g_entropy_path = oj(entropy_path, 'gesture_entropy_full_item.xlsx')
g_entropy = pd.read_excel(g_entropy_path)
s_entropy_path = oj(entropy_path, 'speech_entropy_full_item.xlsx')
s_entropy = pd.read_excel(s_entropy_path)

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

# calculate N400 different wave

component_list = [(300, 500)]
component_chan = {'LC': ['C1', 'C3', 'C5', 'CP1', 'CP3', 'CP5']}

condition_list = np.arange(9)
eeg_data = np.zeros([len(component_chan), len(stimulus_list), len(condition_list)])
for comp_index, component in enumerate(component_list):
    time_index = [int(component[k] / 2 + 100) for k in np.arange(2)]
    time_num_index = np.arange(time_index[0], time_index[1], 1)
    for chan_index, chan in enumerate(component_chan):
        cond_index = 0
        for dp_index, dpn in enumerate(dp_cond):
            for ip_index, ipn in enumerate(ip_cond):
                keep_data = np.empty([0, len(component_chan[chan]), time_point])
                stim_label = np.empty([0])
                for sub_index, sub in enumerate(sub_list):
                    diff_data = np.zeros([2, len(stimulus_list), len(component_chan[chan]), time_point])
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

                stim_data = np.zeros([19, len(component_chan[chan]), time_point])

                for stim_index, stim_name in enumerate(stimulus_list):
                    stim_diff = keep_data[stim_label == stim_index]
                    stim_data[stim_index] = stim_diff.mean(axis=0)

                target_data = stim_data[:, :, time_num_index].mean(axis=(1, 2))
                eeg_data[chan_index, :, cond_index] = target_data
                cond_index += 1

flat_eeg = eeg_data[0].flatten() * (10 ** 6)

sem_data = pd.DataFrame({'mi': flat_mi,
                         'ig_ges': flat_ig_ges,
                         'ig_spe': flat_ig_spe,
                         'ig_unit': flat_ig_unit,
                         'ges_cond': flat_ges_cond,
                         'spe_cond': flat_spe_cond,
                         'all_cond': flat_cond,
                         'lex_cond': flat_lexical,
                         'ges_lex': flat_ges_lex,
                         'spe_lex': flat_spe_lex,
                         'rt_diff': flat_rt,
                         'eeg_diff': flat_eeg})


# constructing mediation model
med_p = Process(data=sem_data, model=4, x='ig_unit', y='rt_diff', m=['ig_ges', 'ig_spe'])
print(med_p.summary())

# constructing moderated mediation model
p = Process(data=sem_data, model=10, x='mi', y='eeg_diff', m=['ig_ges', 'ig_spe'], w='ges_lex', z='spe_lex')
print(p.summary())

# constructing validation model
val_p = Process(data=sem_data, model=10, x='ig_unit', y='eeg_diff', m=['ig_ges', 'ig_spe'], w='ges_lex', z='spe_lex')
val_med_p = Process(data=sem_data, model=4, x='ig_unit', y='eeg_diff', m=['ig_ges', 'ig_spe'])

# this part is plotting part
plot_name = ['rt_diff', 'ig_unit', 'eeg_diff']
target_metric = 'N400'
# now we should plot conditional direct effect in moderator
fig, ax = plt.subplots(1, 1)

color_list = ['#ABD3E1', '#427AB2', '#ABD3E1', '#427AB2']
shape_list = ['^', '^', 'D', 'D']
marker_size = [80, 80, 45, 45]
dash_color = '#92B4C8'
scatter_list = []

cond_pick = [0, 2, 1, 3]
for cond_index, cond in enumerate(np.arange(4)):

    coordinate = p.direct_model._estimation_results['betas'][cond_pick[cond_index]]
    bootLL = p.direct_model._estimation_results['llci'][cond_pick[cond_index]]
    bootUL = p.direct_model._estimation_results['ulci'][cond_pick[cond_index]]

    ax.plot((cond, cond), (bootLL, bootUL), color='black', zorder=0)
    scatter = ax.scatter(cond, coordinate, color=color_list[cond_index],
               edgecolors='black', s=marker_size[cond_index], zorder=1, marker=shape_list[cond_index])
    # line = ax.errorbar(x[cond_index], coordinate, (bootUL - bootLL) / 2, color=color_list[cond_index])
    scatter_list.append(scatter)

# indirect_index = np.concatenate((np.arange(6, 10, 1), np.arange(12, 16, 1)))
for cond_index, cond in enumerate(np.arange(8, 12, 1)):
    coordinate = p.indirect_model.estimation_results['effect'][cond_pick[cond_index]]
    bootLL = p.indirect_model.estimation_results['llci'][cond_pick[cond_index]]
    bootUL = p.indirect_model.estimation_results['ulci'][cond_pick[cond_index]]

    ax.plot((cond, cond), (bootLL, bootUL), color='black', zorder=0)
    scatter = ax.scatter(cond, coordinate, color=color_list[cond_index],
                         edgecolors='black', s=marker_size[cond_index], zorder=1, marker=shape_list[cond_index])
    # line = ax.errorbar(x[cond_index], coordinate, (bootUL - bootLL) / 2, color=color_list[cond_index])
    # scatter_list.append(scatter)

for cond_index, cond in enumerate(np.arange(16, 20, 1)):
    coordinate = p.indirect_model.estimation_results['effect'][cond_pick[cond_index] + 4]
    bootLL = p.indirect_model.estimation_results['llci'][cond_pick[cond_index] + 4]
    bootUL = p.indirect_model.estimation_results['ulci'][cond_pick[cond_index] + 4]

    ax.plot((cond, cond), (bootLL, bootUL), color='black', zorder=0)
    scatter = ax.scatter(cond, coordinate, color=color_list[cond_index],
                         edgecolors='black', s=marker_size[cond_index], zorder=1, marker=shape_list[cond_index])
    # line = ax.errorbar(x[cond_index], coordinate, (bootUL - bootLL) / 2, color=color_list[cond_index])
    # scatter_list.append(scatter)

ax.plot([-1, 20], [0, 0], linestyle='--', color='black', linewidth=1)
ax.set_xlim(-1, 21)
ax.set_xticklabels(['Direct effect',
                    'Gain(gesture)',
                    'Gain(speech)'], fontdict={'fontsize': 12, 'weight': 'bold'})
ax.set_xticks(ticks=[1.5, 9.5, 18.5])
# ax.set_yticks(ticks=np.arange(-45, 16, 5))
# ax.set_ylim(-0.6, 2.5)  # rt: -55, 30; eeg: -0.6, 2.5
ax.yaxis.set_major_locator(plt.MaxNLocator(7))
# ax.set_title('Direct effects of MI on unit information gain')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='y', labelsize=18, labelfontfamily={'weight': 'bold'})

legend_plot1 = plt.scatter([100], [0], color='white', edgecolors='black', s=marker_size[0], marker=shape_list[0])
legend_plot2 = plt.scatter([101], [0], color='white', edgecolors='black',  s=marker_size[2], marker=shape_list[2])

legend_plot3 = plt.scatter([102], [0], color=color_list[2], s=marker_size[0], marker='o')
legend_plot4 = plt.scatter([103], [0], color=color_list[3], s=marker_size[0], marker='o')
legend_list = [legend_plot1, legend_plot2, legend_plot3, legend_plot4]

leg = fig.legend(legend_list, ['gesture pre-lexical', 'gesture post-lexical',
                               'speech pre-lexical', 'speech post-lexical'], fontsize=12,
                 loc=[0.52, 0.6], frameon=False, ncol=1, facecolor='white')

# fig.savefig(oj(fig_path, f'moderated_mediation_modal_of_{target_metric}.png'), dpi=600)
fig.savefig(oj(fig_path, f'inverse_moderated_mediation_modal_of_eeg_diff.png'), dpi=600)
plt.show()


# now we should plot INDEX OF PARTIAL MODERATED MEDIATION
fig, ax = plt.subplots(1, 1)

x = [1, 2, 4, 5]
shape_list = ['^', '^', 'D', 'D']
marker_size = [100, 100, 100, 100]
color_list = ['#8CC5BE', '#6A8EC9', '#8CC5BE', '#6A8EC9']
dash_color = '#92B4C8'
line_list = []
gesture_index = [2, 0, 3, 1]
# gesture_index = [4, 6, 5, 7]

indirect_model = p.indirect_model
summary_str = indirect_model.__str__()
str_list = summary_str.split(' ')
keep_str_list = [s for s in str_list if len(s) != 0]
index_list = [float(keep_str_list[91]), float(keep_str_list[97]), float(keep_str_list[103]), float(keep_str_list[109])]
boot_ll_list = [float(keep_str_list[93]), float(keep_str_list[99]), float(keep_str_list[105]), float(keep_str_list[111])]
boot_ul_list = [float(keep_str_list[94]), float(keep_str_list[100]), float(keep_str_list[106]), float(keep_str_list[112])]

for plt_index, cond_index in enumerate(gesture_index):
    coordinate = index_list[cond_index]

    bootLL = boot_ll_list[cond_index]
    bootUL = boot_ul_list[cond_index]

    scatter = ax.scatter(x[plt_index], coordinate, edgecolors='black', color=color_list[plt_index], s=200,
               marker='o', zorder=1, linewidth=0)
    # line = ax.errorbar(x[plt_index], coordinate, (bootUL - bootLL) / 2, color=color_list[plt_index])
    ax.plot((x[plt_index], x[plt_index]), (bootLL, bootUL), color=color_list[plt_index], zorder=0,
            linewidth=7)
    line_list.append(scatter)

ax.plot([0, 6], [0, 0], linestyle='--', color='black', linewidth=1)

ax.set_xlim(0, 6)
ax.set_xticklabels(['Gain(gesture)', 'Gain(speech)'],
                   fontdict={'fontsize': 12, 'weight': 'bold'})
ax.set_xticks(ticks=[1.5, 4.5])
# ax.set_ylim([-0.5, 0.5])  # rt: -30, 35; eeg: -0.8, 1.2
# ax.set_ylim([-10, 10])
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
ax.set_title('Index of partial moderated mediation')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='y', labelsize=18, labelfontfamily={'weight': 'bold'})


leg = fig.legend([line_list[0], line_list[1]], ['gesture lexical', 'speech lexical'], fontsize=12,
                 loc=[0.6, 0.75], frameon=False, ncol=1, facecolor='white')
# fig.savefig(oj(fig_path, f'{target_metric}_moderated_effect.png'), dpi=600)
fig.savefig(oj(fig_path, f'inverse_eeg_diff_moderated_effect.png'), dpi=600)
plt.show()


# plot the correlation figure between rt diff, N400 and unit information gain
plt_color = '#e4bad5'
method = 'spearman'

compare_list = ['rt_diff', 'eeg_diff']

label_list = ['Gain(unit)', 'N400 different wave(uv)']
fig, ax = plt.subplots(1, 1)
sns.regplot(sem_data, x=compare_list[0], y=compare_list[1], ax=ax, color=plt_color)
corr_stats = pg.corr(sem_data[compare_list[0]], sem_data[compare_list[1]], method=method)
ax.set_xlabel(xlabel=label_list[0], fontdict={'fontsize': 22, 'weight': 'bold'}, labelpad=10)
ax.set_ylabel(ylabel=label_list[1], fontdict={'fontsize': 22, 'weight': 'bold'})
ticks = ax.get_xticks()

for label in ax.get_xticklabels():
    label.set_fontweight('bold')
    label.set_fontsize(18)


for label in ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontsize(18)

ax.tick_params(axis='both', width=3, length=7)
ax.yaxis.set_major_locator(plt.MaxNLocator(7))
ax.xaxis.set_major_locator(plt.MaxNLocator(7))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
fig.tight_layout()
fig.subplots_adjust(left=0.1, right=0.98, wspace=0.3, bottom=0.13)
fig.savefig(oj(fig_path, f'{compare_list[0]}&{compare_list[1]}_regression_figure.png'), dpi=600)
plt.show()


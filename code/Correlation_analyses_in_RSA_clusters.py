# -*- coding = utf-8 -*-
# @Time : 30/09/2024 21.20
# @Author : yangjw
# @Site : 
# @File : Correlation_analyses_in_RSA_clusters.py
# @Software : PyCharm
# @contact: jwyang9826@gmail.com



import mne
import numpy as np
import pandas as pd
import scipy.io as sio
import os
from jw_function.supply_function import create_dir
from jw_function.bhv_function import find_useless_trial
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import fdrcorrection
from skimage.measure import label
from neurora.stuff import smooth_1d
import matplotlib
from os.path import join as oj

work_directory = os.getcwd()
project_path = work_directory + '/'
fig_path = project_path + '/figure/'
behavior_path = oj(work_directory, 'behavioral_information')

excel_path = oj(behavior_path, 'stimuli_information_update.xls')
stim_info = pd.read_excel(excel_path)

entropy_path = project_path + '/entropy/'

g_entropy_path = entropy_path + 'gesture_entropy_full_item.xlsx'
g_entropy = pd.read_excel(g_entropy_path, index_col=0)
s_entropy_path = entropy_path + 'speech_entropy_full_item.xlsx'
s_entropy = pd.read_excel(s_entropy_path, index_col=0)


new_mi = pd.read_excel(entropy_path + 'block1_infor.xls')

new_stim_info = pd.read_excel(project_path + '/stimili_information_update_20.xls')

data_path = '/storage/work/yangjw/ddm_project/'
new_path = data_path + 'gesture_raw_block1_delaud3pair_delmisserror_z_bysubj_2z.sav'
new_data = pd.read_spss(new_path)

stim_name = [stim_str.split('.avi')[0] for stim_str in new_stim_info.iloc[:, 0]]
response_name = ['f', 'm']

trial_dp = np.array([stim_name.index(trial.split('.avi')[0].split('-')[1]) for trial in new_data['vid']])
trial_ip = np.array([stim_name.index(trial.split('.wav')[0].split('-')[1]) for trial in new_data['aud']])
response = np.array([response_name.index(trial.split('.wav')[0].split('-')[0]) for trial in new_data['aud']])

new_data['dp_stim'] = trial_dp
new_data['ip_stim'] = trial_ip
new_data['response'] = response
new_data = new_data.rename(columns={'reac_rt': 'rt'})

create_dir(fig_path)

all_sub = np.arange(1, 40)
miss_sub = [7, 8, 10, 17, 20, 21, 23, 24, 25]
sub_list = [sub for sub in all_sub if sub not in miss_sub]

dp_cond = np.arange(1, 4)
ip_cond = np.arange(1, 4)
congru_cond = np.arange(1, 3)
condition_list = np.arange(len(dp_cond) * len(ip_cond))

dp_cond_name = ['BD', 'DP', 'AD']
ip_cond_name = ['BI', 'IP', 'AI']
all_cond_name = [dpn + '&' + ipn for dpn in dp_cond_name for ipn in ip_cond_name]

chan_num = 42
time_point = 600

# first part: correlation between difference waves and entropy loss
# loading entropy loss
entropy_loss = pd.read_csv(entropy_path + 'entropy_loss.csv', index_col=0)
entropy_loss = entropy_loss.T
mi_path = entropy_path + 'mutual_information_full_item.xlsx'
mi_entropy = pd.read_excel(mi_path, index_col=0)

stimuli_list = [sn.split('.avi')[0] for sn in stim_info.iloc[:, 0]]

index_list = [0]
index_list.extend(all_cond_name.copy())
mi_entropy.columns = np.array(index_list)
mi_entropy.index = pd.Index(stimuli_list)

# loading significant time window of different cluster
rdm_label = ['congruence RDM', 'gesture RDM', 'speech RDM'] # mark the sequence of factor
all_rdm_stats = np.load(project_path + '/full factorial RSA sig cluster time points onw-sided.npy')
all_beta = np.load(project_path + '/full_factorial_RSA_all_beta.npy')
sig_time_win = np.array(all_rdm_stats[0, :], dtype='bool')
sig_label = label(sig_time_win)

# now we can calculate erp in all stimulus
cond_index = 0
condition_array = np.empty(shape=[0])
data_array = np.empty(shape=[0, chan_num, time_point])
stim_label = np.empty(shape=[0])
ip_stim_label = np.empty(shape=[0])
sub_label = np.empty(shape=[0])
congru_label = np.empty(shape=[0])
for dp_index, d_cond in enumerate(dp_cond):
    for ip_index, i_cond in enumerate(ip_cond):
        for sub_index, sub in enumerate(sub_list):
            for con_index, congru in enumerate(congru_cond):
                cond_folder = project_path + '/' + str(d_cond) + '_' + str(i_cond) + '_' + str(congru) + '/'
                event_mat = sio.loadmat(cond_folder + str(sub) + '.mat')
                delete_id, new_event_df = find_useless_trial(event_mat, stim_info)
                sub_epochs = mne.read_epochs_eeglab(cond_folder + str(sub) + '.set')
                # sub_epochs.drop_channels(drop_channel)
                # sub_epochs.drop(delete_id)
                sub_epochs.apply_baseline(baseline=(-0.2, 0))

                data_array = np.concatenate((data_array, sub_epochs._data.copy()), axis=0)
                stim_label = np.concatenate((stim_label, new_event_df['dp_stim']))
                ip_stim_label = np.concatenate((ip_stim_label, new_event_df['ip_stim']))
                condition_array = np.concatenate((condition_array,
                                                  [cond_index for i in np.arange(len(sub_epochs))]))
                sub_label = np.concatenate((sub_label, [sub_index for i in np.arange(len(sub_epochs))]))
                congru_label = np.concatenate((congru_label, [con_index for i in np.arange(len(sub_epochs))]))

        cond_index += 1


np.save(project_path + '/eeg array for rsa correlation.npy', data_array)
np.save(project_path + '/sub_label for rsa correlation.npy', sub_label)
np.save(project_path + '/stim_label for rsa correlation.npy', stim_label)
np.save(project_path + '/ip_stim_label for rsa correlation.npy', ip_stim_label)
np.save(project_path + '/congru_label for rsa correlation.npy', congru_label)
np.save(project_path + '/condition_array for rsa correlation.npy', condition_array)

data_array = np.load(project_path + '/eeg array for rsa correlation.npy')
sub_label = np.load(project_path + '/sub_label for rsa correlation.npy')
stim_label = np.load(project_path + '/stim_label for rsa correlation.npy')
ip_stim_label = np.load(project_path + '/ip_stim_label for rsa correlation.npy')
congru_label = np.load(project_path + '/congru_label for rsa correlation.npy')
condition_array = np.load(project_path + '/condition_array for rsa correlation.npy')

stimulus_list = [stim_str.split('.avi')[0] for stim_str in stim_info.iloc[:, 0]]

mode_epochs = mne.read_epochs_eeglab(project_path + '/1_1_1/1.set')

# setting the cluster we want to calculate correlation
sig_time_win = np.array(all_rdm_stats[0, :], dtype='bool')
sig_label = label(sig_time_win)
sig_time_win = (sig_label == 1)

# checking the standard error of beta
mean_beta = all_beta[0, sig_time_win].mean(axis=0)
mean = mean_beta.mean(axis=0)
se = mean_beta.std(axis=0)/29

gesture_state = [dp_index for dp_index in np.arange(3) for ip_index in np.arange(3) for stim in np.arange(19)]
speech_state = [ip_index for dp_index in np.arange(3) for ip_index in np.arange(3) for stim in np.arange(19)]

# this part we calculate correlation between different wave and information-theoretic indicators
target_indicator = mi_entropy  # entropy_loss (i.e. Gain(unit)) or mi_entropy
chan_r = np.zeros([chan_num])
chan_p = np.zeros([chan_num])
method = 'spearman'
all_eeg_data_flat = np.zeros([chan_num, len(condition_list) * len(stimulus_list)])
chan_r_list = np.empty([0])
for chan in np.arange(chan_num):
    cond_index = 0

    mi_flat = np.empty([0])
    eeg_data_flat = np.empty([0])
    condition_flat = np.empty([0])
    stim_flat = np.empty([0])
    for dp_index, d_cond in enumerate(dp_cond):
        for ip_index, i_cond in enumerate(ip_cond):
            for stim_index, stim_name in enumerate(stim_info.iloc[:, 0]):
                real_sn = stim_name.split('.')[0]
                congru_sub_flat = np.empty([0])
                incongru_sub_flat = np.empty([0])
                for sub_index, sub in enumerate(sub_list):

                    congru_eeg_data = np.average(data_array[(condition_array == cond_index)
                                                            & (stim_label == stim_index) & (sub_label == sub_index)
                                                            & (congru_label == 0)][:, chan, sig_time_win])
                    incongru_eeg_data = np.average(data_array[(condition_array == cond_index)
                                                              & (stim_label == stim_index) & (sub_label == sub_index)
                                                              & (congru_label == 1)][:, chan, sig_time_win])

                    if ~np.isnan(congru_eeg_data) and ~np.isnan(incongru_eeg_data):
                        congru_sub_flat = np.concatenate((congru_sub_flat, [congru_eeg_data]))
                        incongru_sub_flat = np.concatenate((incongru_sub_flat, [incongru_eeg_data]))

                # calculate different wave amplitude
                single_eeg_data = incongru_sub_flat.mean(axis=0) - congru_sub_flat.mean(axis=0)
                mi_flat = np.concatenate((mi_flat, [target_indicator.loc[real_sn, all_cond_name[cond_index]]]))
                eeg_data_flat = np.concatenate((eeg_data_flat, [single_eeg_data]))
                condition_flat = np.concatenate((condition_flat, [cond_index]))
                stim_flat = np.concatenate((stim_flat, [stim_index]))

            cond_index += 1

    all_eeg_data_flat[chan] = eeg_data_flat

    print(chan)
    corr_stats = pg.corr(mi_flat, eeg_data_flat, method=method)
    chan_r[chan] = corr_stats.loc[method, 'r']
    chan_p[chan] = corr_stats.loc[method, 'p-val']

    chan_r_list = np.concatenate((chan_r_list, [corr_stats.loc[method, 'r']]))

    if abs(corr_stats.loc[method, 'r']) == np.max(abs(chan_r_list)):
        plot_eeg = eeg_data_flat * (10 ** 6)
        plot_infor = mi_flat
        max_corr = corr_stats.loc[method, 'r']
        min_p = corr_stats.loc[method, 'p-val']
        print(f"max r: {corr_stats.loc[method, 'r']}, min p: {corr_stats.loc[method, 'p-val']}")

# fdr correction
p_bool, corrected_p = fdrcorrection(chan_p)
effect_chan = chan_r[p_bool]
min_chan_r = np.min(effect_chan)
max_chan_r = np.max(effect_chan)
max_chan_p = np.max(corrected_p[p_bool])

fig, ax = plt.subplots(1, 1)
mi_color = '#CC011F'
# unit gain: #e4b9d4
# mi: #CC011F

plot_data = pd.DataFrame({'x': plot_infor, 'y': plot_eeg})
sns.regplot(data=plot_data, x='x', y='y', color=mi_color, ax=ax)

# ax = plotting_correlation_result(plot_infor, plot_eeg, resolution=1000, color=mi_color, alpha=0.7, ax=ax)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(2.5)
ax.spines['left'].set_linewidth(2.5)

ax.set_xlabel('Mutual information', fontdict={'fontsize': 20, 'weight': 'bold'}, labelpad=10)
ax.set_ylabel('Different wave (uv)', fontdict={'fontsize': 20, 'weight': 'bold'}, labelpad=10)
ax.tick_params(labelsize=24, labelfontfamily={'fontweight': 'bold'})
fig.tight_layout()
fig.subplots_adjust(bottom=0.2, top=0.95)
fig.savefig(fig_path + '/mi first cluster max correlation fig.png', dpi=600)
plt.show()

# plotting topography figure of correlation
fig, ax = plt.subplots(1, 1)
colormap = 'coolwarm'
channel_camp = matplotlib.colormaps[colormap]

true_list = np.array([True for i in np.arange(len(mode_epochs.ch_names))])
mask_para = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                 linewidth=0, markersize=0)
im = mne.viz.plot_topomap(chan_r, mode_epochs.info, res=600, cmap=channel_camp, axes=ax,
                          names=None, mask=true_list, mask_params=mask_para,
                          show=False, vlim=[-0.5, 0.5])

p_bool, corrected_p = fdrcorrection(chan_p)
max_bool = np.array([i != np.argmax(chan_r) for i in np.arange(len(chan_r))])

all_position = np.zeros([len(mode_epochs.ch_names), 3])
for chan in np.arange(len(mode_epochs.ch_names)):
    all_position[chan] = mode_epochs.info['dig'][chan + 3]['r']

# rdm_color = ['#CC011F', '#8CC5BE', '#6A8EC9']
color = 'black'
x, y = all_position[p_bool * max_bool, :2].T * 0.8

ax.scatter(x, y, s=80, c=color, marker='o', edgecolor='k')

ax.scatter(all_position[~max_bool, 0] * 0.8, all_position[~max_bool, 1] * 0.8, s=500, c='black', marker='*', edgecolor='k')

cax = fig.add_axes([0.85, 0.2, 0.03, 0.6])
mappable = im[0]  # 获取地形图的映射对象
cb = plt.colorbar(mappable, cax=cax)
cb.cmap = colormap

plt.savefig(fig_path + '/mi_correlation_topography', dpi=600)
plt.show()


# next part we can conduct correlation between gesture entropy and erp
rdm_label = ['congruence RDM', 'gesture RDM', 'speech RDM']

# set parameter for checking cluster you want
# dp_time_win = np.array(all_rdm_stats[2, :], dtype='bool')
# dp_label = label(dp_time_win)
# dp_time_win = (dp_label == 2)
# dp_time_win[np.arange(150, 600, 1)] = False
# dp_time_win[np.arange(225, 600, 1)] = False
# dp_time_win[np.arange(0, 175, 1)] = False


data_array = np.load(project_path + '/eeg array for rsa correlation.npy')
sub_label = np.load(project_path + '/sub_label for rsa correlation.npy')
stim_label = np.load(project_path + '/stim_label for rsa correlation.npy')
ip_stim_label = np.load(project_path + '/ip_stim_label for rsa correlation.npy')
congru_label = np.load(project_path + '/congru_label for rsa correlation.npy')
condition_array = np.load(project_path + '/condition_array for rsa correlation.npy')

dp_cond_label = np.array([int(i / 3) for i in condition_array])
ip_cond_label = np.array([int(i % 3) for i in condition_array])

gesture_entropy_x = pd.concat((g_entropy.loc[:, 0.75], g_entropy.loc[:, 0.75], g_entropy.loc[:, 0.75],
                               g_entropy.loc[:, 1.0], g_entropy.loc[:, 1.0], g_entropy.loc[:, 1.0],
                               g_entropy.loc[:, 1.25], g_entropy.loc[:, 1.25], g_entropy.loc[:, 1.25]), axis=1)
gesture_entropy_x.columns = pd.Index(all_cond_name)
speech_entropy_x = pd.concat((s_entropy, s_entropy, s_entropy), axis=1)
speech_entropy_x.columns = pd.Index(all_cond_name)

sig_time_win = np.array(all_rdm_stats[1, :], dtype='bool')
sig_label = label(sig_time_win)
dp_time_win = (sig_label == 1)
dp_time_win[151::] = False
# dp_time_win[np.arange(225, 600, 1)] = False
# dp_time_win[np.arange(0, 175, 1)] = False

gesture_state = [dp_index for dp_index in np.arange(3) for ip_index in np.arange(3) for stim in np.arange(19)]
speech_state = [ip_index for dp_index in np.arange(3) for ip_index in np.arange(3) for stim in np.arange(19)]

target_state = gesture_state
target_entropy_x = gesture_entropy_x

c_chan_r = np.zeros([chan_num])
ic_chan_r = np.zeros([chan_num])
c_chan_p = np.zeros([chan_num])
ic_chan_p = np.zeros([chan_num])
all_chan_r = np.zeros([chan_num])
all_chan_p = np.zeros([chan_num])
method = 'spearman'
chan_r_list = np.empty([0])
all_eeg_data_flat = np.zeros([3, chan_num, len(condition_list) * len(stimulus_list)])
# first element is congruent, second one is incongruent, and last one is all congruency condition data
for chan in np.arange(chan_num):
    cond_index = 0

    entropy_flat = np.empty([0])
    full_eeg_flat = np.empty([0])
    congru_eeg_data_flat = np.empty([0])
    incongru_eeg_data_flat = np.empty([0])
    condition_flat = np.empty([0])
    stim_flat = np.empty([0])
    for dp_index, d_cond in enumerate(dp_cond):
        for ip_index, i_cond in enumerate(ip_cond):
            for stim_index, stim_name in enumerate(stim_info.iloc[:, 0]):
                real_sn = stim_name.split('.')[0]
                eeg_sub_flat = np.empty([0])
                congru_sub_flat = np.empty([0])
                incongru_sub_flat = np.empty([0])
                for sub_index, sub in enumerate(sub_list):

                    congru_eeg_data = np.average(data_array[(condition_array == cond_index)
                                                            & (stim_label == stim_index) & (sub_label == sub_index)
                                                            & (congru_label == 0)][:, chan, dp_time_win])
                    incongru_eeg_data = np.average(data_array[(condition_array == cond_index)
                                                              & (stim_label == stim_index) & (sub_label == sub_index)
                                                              & (congru_label == 1)][:, chan, dp_time_win])
                    full_eeg_data = np.average(data_array[(condition_array == cond_index)
                                                          & (stim_label == stim_index) &
                                                          (sub_label == sub_index)][:, chan, dp_time_win])

                    if ~np.isnan(congru_eeg_data) and ~np.isnan(incongru_eeg_data):
                        congru_sub_flat = np.concatenate((congru_sub_flat, [congru_eeg_data]))
                        incongru_sub_flat = np.concatenate((incongru_sub_flat, [incongru_eeg_data]))

                    if ~np.isnan(full_eeg_data):
                        eeg_sub_flat = np.concatenate((eeg_sub_flat, [full_eeg_data]))

                entropy_flat = np.concatenate((entropy_flat, [target_entropy_x.loc[
                                                                  real_sn, all_cond_name[cond_index]]]))
                congru_eeg_data_flat = np.concatenate((congru_eeg_data_flat, [congru_sub_flat.mean(axis=0)]))
                incongru_eeg_data_flat = np.concatenate((incongru_eeg_data_flat, [incongru_sub_flat.mean(axis=0)]))
                full_eeg_flat = np.concatenate((full_eeg_flat, [eeg_sub_flat.mean(axis=0)]))
                condition_flat = np.concatenate((condition_flat, [cond_index]))
                stim_flat = np.concatenate((stim_flat, [stim_index]))

            cond_index += 1

    all_eeg_data_flat[0, chan] = congru_eeg_data_flat
    all_eeg_data_flat[1, chan] = incongru_eeg_data_flat
    all_eeg_data_flat[2, chan] = full_eeg_flat

    print(chan)

    full_corr_stats = pg.corr(entropy_flat, full_eeg_flat, method=method)

    all_chan_r[chan] = full_corr_stats.loc[method, 'r']
    all_chan_p[chan] = full_corr_stats.loc[method, 'p-val']

    chan_r_list = np.concatenate((chan_r_list, [full_corr_stats.loc[method, 'r']]))

    if abs(full_corr_stats.loc[method, 'r']) == np.max(abs(chan_r_list)):
        plot_eeg = full_eeg_flat * (10 ** 6)
        plot_infor = entropy_flat
        max_corr = full_corr_stats.loc[method, 'r']
        min_p = full_corr_stats.loc[method, 'p-val']
        print(f"max r: {full_corr_stats.loc[method, 'r']}, min p: {full_corr_stats.loc[method, 'p-val']}")


# rdm_color = ['#CC011F', '#8CC5BE', '#6A8EC9']
fig, ax = plt.subplots(1, 1)
mi_color = '#8CC5BE'  # ['#CC011F', '#8CC5BE', '#6A8EC9']
# plot Cz channel correlation figure in sig cluster
plot_data = pd.DataFrame({'x': plot_infor, 'y': plot_eeg})
sns.regplot(data=plot_data, x='x', y='y', color=mi_color, ax=ax)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(2.5)
ax.spines['left'].set_linewidth(2.5)

ax.set_xlabel('Speech entropy', fontdict={'fontsize': 20, 'weight': 'bold'}, labelpad=10)
ax.set_ylabel('Amplitude (uv)', fontdict={'fontsize': 20, 'weight': 'bold'}, labelpad=10)
ax.tick_params(labelsize=24, labelfontfamily={'fontweight': 'bold'})
fig.tight_layout()
fig.subplots_adjust(bottom=0.2, top=0.95)
fig.savefig(fig_path + '/gesture first cluster max correlation fig.png', dpi=600)
plt.show()


mode_epochs = mne.read_epochs_eeglab(project_path + '/1_1_1/1.set')

target_r = all_chan_r
target_p = all_chan_p
fig, ax = plt.subplots(1, 1)
channel_camp = matplotlib.colormaps[colormap]

true_list = np.array([True for i in np.arange(len(mode_epochs.ch_names))])
mask_para = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                 linewidth=0, markersize=0)
im = mne.viz.plot_topomap(target_r, mode_epochs.info, res=600, cmap=channel_camp, axes=ax,
                          names=None, mask=true_list, mask_params=mask_para,
                          show=False, vlim=[-0.5, 0.5])

p_bool, corrected_p = fdrcorrection(target_p)
max_bool = np.array([i != np.argmin(target_r) for i in np.arange(len(target_r))])

all_position = np.zeros([len(mode_epochs.ch_names), 3])
for chan in np.arange(len(mode_epochs.ch_names)):
    all_position[chan] = mode_epochs.info['dig'][chan + 3]['r']

# rdm_color = ['#CC011F', '#8CC5BE', '#6A8EC9']
color = 'black'
x, y = all_position[p_bool * max_bool, :2].T * 0.8

ax.scatter(x, y, s=80, c=color, marker='o', edgecolor='k')

ax.scatter(all_position[~max_bool, 0] * 0.8, all_position[~max_bool, 1] * 0.8, s=500, c='black', marker='*',
           edgecolor='k')

cax = fig.add_axes([0.85, 0.2, 0.03, 0.6])
mappable = im[0]  # 获取地形图的映射对象
cb = plt.colorbar(mappable, cax=cax)
cb.cmap = colormap

plt.savefig(fig_path +
            'Gesture_entropy_topography_figure_of_correlation.png', dpi=600)
plt.show()

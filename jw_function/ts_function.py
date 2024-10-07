# -*- coding = utf-8 -*-
# @Time : 20/01/2024 20.45
# @Author : yangjw
# @Site : 
# @File : ts_function.py
# @Software : PyCharm
# @contact: jwyang9826@gmail.com


import pandas as pd
from mne.stats import combine_adjacency, permutation_cluster_1samp_test
import numpy as np
from scipy.stats import stats
from mne.channels import find_ch_adjacency
from statsmodels.stats.multitest import fdrcorrection
import pingouin as pg
from jw_function.plot_function import show_progressbar
from mne.stats.cluster_level import _find_clusters


def spatio_temporal_t_cluster(t_data, epochs, p_threshold=0.05, cluster_p=0.05, minium_time=10, n_permutation=1024,
                              tail=0, cls_type='all', ch_type='eeg', output_type='indices', fdr=True):

    """

    Args:
        t_data: array. It contains t value across all participant, channel, and time point.
                shape: [sub_num, channel_num, time_point]
        p_threshold: alpha level of single point
        cluster_p: p threshold of cluster
        minium_time: the min time point length of effective cluster, exclude all clusters shorter than 10ms
        epochs: Epoch object in mne, which can provide information to creating adjacency object
        n_permutation: permutation times
        tail: Default is 0, meaning two-tailed test. 1 means one-tailed test and greater, -1 is less.
        ch_type: Default is eeg. You can change it if your data is meg.
        output_type: Default is 'indices', which will create list of arrays, containing indices of channel and
                     time points of clusters. If you set 'mask', it will contain bool indices.
        cls_type: Default is 'all', meaning out put all effective cluster. If you send 'max', it will only return
                  max cluster
        fdr: Default is True. Determine if you conduct fdr correction

    Returns:

    """

    sub_num, chan_num, time_point = t_data.shape

    t_threshold = stats.distributions.t.ppf(1 - p_threshold / 2, df=sub_num - 1)

    if tail == -1:
        t_threshold *= -1
    # getting significant threshold of t value
    adjacency, ch_name = find_ch_adjacency(epochs.info, ch_type=ch_type)
    # calculating sparse matrix
    com_adj = combine_adjacency(adjacency, time_point)
    # transform to 2D adjacency

    cls_stats = permutation_cluster_1samp_test(
                t_data,
                n_permutations=n_permutation,
                threshold=t_threshold,
                tail=tail,
                adjacency=com_adj,
                out_type=output_type,
                verbose=True)
    # conducting spatiotemporal cluster-based permutation

    if fdr:
        # doing fdr correction and save significant cluster
        effect_bool, correct_p = fdrcorrection(cls_stats[2], alpha=cluster_p)
        effect_index = [i for i in np.arange(len(cls_stats[1])) if effect_bool[i]]
    else:
        effect_bool = cls_stats[2] <= cluster_p
        effect_index = [i for i in np.arange(len(cls_stats[1])) if effect_bool[i]]

    # exclude short cluster
    effect_cluster = []
    cls_length_list = []
    for e_index, e in enumerate(effect_index):

        single_cls = cls_stats[1][e]
        chan_cls = single_cls[0]
        time_cls = single_cls[1]
        cls_length = np.max(time_cls) - np.min(time_cls)

        # and we need to exclude some channel if their data is shorter than 10ms
        sig_chan = pd.value_counts(chan_cls).index
        exclude_chan = []
        for chan_index in sig_chan:
            chan_time = time_cls[chan_cls == chan_index]
            if len(chan_time) < minium_time:
                exclude_chan.append(chan_index)
                print(f'excluding channel: {ch_name[chan_index]} in cluster {e_index}')
        final_chan = chan_cls[np.array([single_chan not in exclude_chan for single_chan in chan_cls])]
        final_time = time_cls[np.array([single_chan not in exclude_chan for single_chan in chan_cls])]
        final_cls = [final_chan, final_time]

        if cls_length > minium_time:
            effect_cluster.append(final_cls)
            cls_length_list.append(cls_length)

    if cls_type == 'max':
        max_index = np.argmax(cls_length_list)
        final_cluster = [effect_cluster[max_index]]
    else:
        final_cluster = effect_cluster.copy()

    return final_cluster, cls_stats[0], t_threshold


def spatio_temporal_t_cluster_2(t_data, epochs, p_threshold=0.05, cluster_p=0.05, minium_time=10, n_permutation=1000,
                                tail=0, cls_type='all', ch_type='eeg', output_type='indices', fdr=True):

    """

    Args:
        t_data: array. It contains t value across all participant, channel, and time point.
                shape: [sub_num, channel_num, time_point]
        p_threshold: alpha level of single point
        cluster_p: p threshold of cluster
        minium_time: the min time point length of effective cluster, exclude all clusters shorter than 10ms
        epochs: Epoch object in mne, which can provide information to creating adjacency object
        n_permutation: permutation times
        tail: Default is 0, meaning two-tailed test. 1 means one-tailed test and greater, -1 is less.
        ch_type: Default is eeg. You can change it if your data is meg.
        output_type: Default is 'indices', which will create list of arrays, containing indices of channel and
                     time points of clusters. If you set 'mask', it will contain bool indices.
        cls_type: Default is 'all', meaning out put all effective cluster. If you send 'max', it will only return
                  max cluster
        fdr: Default is True. Determine if you conduct fdr correction

    Returns:

    """

    sub_num, chan_num, time_point = t_data.shape

    t_threshold = stats.distributions.t.ppf(1 - p_threshold / 2, df=sub_num - 1)

    if tail == -1:
        t_threshold *= -1
    # getting significant threshold of t value
    adjacency, ch_name = find_ch_adjacency(epochs.info, ch_type=ch_type)
    # calculating sparse matrix
    com_adj = combine_adjacency(adjacency, time_point)
    # transform to 2D adjacency

    cls_stats = permutation_cluster_1samp_test(
                t_data,
                n_permutations=10,
                threshold=t_threshold,
                tail=tail,
                adjacency=com_adj,
                out_type=output_type,
                verbose=True)
    # conducting spatiotemporal cluster-based permutation

    length_cls = []
    time_length_list = []
    for single_index, single_cls in enumerate(cls_stats[1]):
        time_cls = single_cls[1]
        time_length = np.max(time_cls) - np.min(time_cls)

        if time_length > minium_time:
            length_cls.append(single_cls)
            time_length_list.append(time_length)

    t_list = []
    p_list = []
    for e_index, single_e_cls in enumerate(length_cls):
        effect_t = t_data[:, single_e_cls[0], single_e_cls[1]]
        average_t = effect_t.mean(axis=1)

        permu_data = np.concatenate((average_t, np.zeros([len(average_t)])))
        real_t_stats = pg.ttest(permu_data[0:len(average_t)], permu_data[len(average_t):len(permu_data)])
        real_t = real_t_stats.loc['T-test', 'T']

        null_distribution = np.zeros([n_permutation])
        for i in np.arange(n_permutation):
            show_progressbar(f'conducting permutation in cluster {e_index + 1}: ', cur=(i/n_permutation) * 100)
            random_data = np.random.permutation(permu_data)
            permu_t_stats = pg.ttest(random_data[0:len(average_t)], random_data[len(average_t):len(permu_data)])
            null_distribution[i] = permu_t_stats.loc['T-test', 'T']

        null_distribution = np.concatenate((null_distribution, [real_t]))
        sort_distribution = list(np.sort(null_distribution))
        loc = sort_distribution.index(real_t)

        if loc < n_permutation/2:
            p = loc / n_permutation
        else:
            p = 1 - loc / n_permutation

        t_list.append(real_t)
        p_list.append(p)

    t_list = np.array(t_list)
    p_list = np.array(p_list)

    if fdr:
        # doing fdr correction and save significant cluster
        effect_bool, correct_p = fdrcorrection(p_list, alpha=cluster_p)
        effect_index = [i for i in np.arange(len(length_cls)) if effect_bool[i]]
    else:
        effect_bool = (p_list <= cluster_p)
        effect_index = [i for i in np.arange(len(length_cls)) if effect_bool[i]]

    if cls_type == 'max':
        max_index = np.argmax(np.array(time_length_list)[effect_bool])
        final_cluster = [length_cls[max_index]]
    else:
        final_cluster = length_cls.copy()

    return final_cluster, cls_stats[0], t_threshold, t_list[effect_bool], p_list[effect_bool],


def spatio_temporal_group_cluster_tfr(t_data, epochs, df, p_threshold=0.001, ch_type='eeg'):
    from scipy import ndimage
    chan_num, freq, time_point = t_data.shape

    t_threshold = stats.distributions.t.ppf(1 - p_threshold / 2, df=df)

    adjacency, ch_name = find_ch_adjacency(epochs.info, ch_type=ch_type)
    # calculating sparse matrix
    com_adj = combine_adjacency(adjacency, freq, time_point)

    output = _find_clusters(t_data.flatten(), t_threshold, tail=0, adjacency=com_adj, max_step=1)

    final_cls = []
    t_list = []
    for cls_index, cls in enumerate(output[0]):
        new_bool = np.zeros(t_data.flatten().shape, dtype='bool')
        new_bool[cls] = True
        new_bool = new_bool.reshape(t_data.shape)

        chan_index = np.empty([0])
        freq_index = np.empty([0])
        time_index = np.empty([0])
        for c_index in cls:
            chan_index = np.concatenate((chan_index, [int(c_index / (freq * time_point))]))
            freq_index = np.concatenate((freq_index, [int(int(c_index % (freq * time_point)) / time_point)]))
            time_index = np.concatenate((time_index, [int(int(c_index % (freq * time_point)) % time_point)]))

        cls_length = int(np.max(time_index) - np.min(time_index))

        if cls_length > 20:
            single = [chan_index, freq_index, time_index, new_bool]
            final_cls.append(single)
            t_list.append(output[1][cls_index])

    return final_cls, t_list

def spatio_temporal_group_cluster_erp(t_data, epochs, df, min_length=5, p_threshold=0.001, ch_type='eeg'):
    from scipy import ndimage
    chan_num, time_point = t_data.shape

    t_threshold = stats.distributions.t.ppf(1 - p_threshold / 2, df=df)

    adjacency, ch_name = find_ch_adjacency(epochs.info, ch_type=ch_type)
    # calculating sparse matrix
    com_adj = combine_adjacency(adjacency, time_point)

    output = _find_clusters(t_data.flatten(), t_threshold, tail=0, adjacency=com_adj, max_step=1)

    final_cls = []
    t_list = []
    for cls_index, cls in enumerate(output[0]):

        new_bool = np.zeros(t_data.flatten().shape, dtype='bool')
        new_bool[cls] = True
        new_bool = new_bool.reshape(t_data.shape)

        chan_index = np.empty([0])
        time_index = np.empty([0])
        for c_index in cls:
            chan_index = np.concatenate((chan_index, [int(c_index / time_point)]))
            time_index = np.concatenate((time_index, [int(c_index % time_point)]))

        cls_length = int(np.max(time_index) - np.min(time_index))

        if cls_length >= min_length:
            single = [chan_index, time_index, new_bool]
            final_cls.append(single)
            t_list.append(output[1][cls_index])

    return final_cls, t_list

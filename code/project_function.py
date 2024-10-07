# -*- coding = utf-8 -*-


import os
import sys
import numpy as np
import matplotlib.colors
from scipy import ndimage
import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from neurora.stuff import show_progressbar, smooth_1d, smooth_2d
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

def create_dir(path):
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')
    else:
        print('目录已存在')


def show_progressbar(string, cur, total=100):
    """
    just for checking progress, copy code in neuroRA's rsa_plot
    :param string:
    :param cur:
    :param total:
    :return:
    """
    percent = '{:.2%}'.format(cur / total)
    sys.stdout.write('\r')
    sys.stdout.write(string + ": [%-100s] %s" % ('=' * int(cur), percent))
    sys.stdout.flush()


def upper_tri(RDM, normalize=True, up_opt='upper'):
    """upper_tri returns the upper triangular index of an RDM

    Args:
        RDM 2Darray: squareform RDM

    Returns:
        1D array: upper triangular vector of the RDM
    """

    if up_opt == 'upper':
        # returns the upper triangle
        m = RDM.shape[0]
        r, c = np.triu_indices(m, 1)
    elif up_opt == 'lower':
        m = RDM.shape[0]
        r, c = np.tril_indices(m, 1)
    else:
        raise ValueError('you should input correct up_opt parameter')

    if not normalize:
        return RDM[r, c]
    else:
        upper = RDM[r, c]
        return upper/np.linalg.norm(upper)


def RDMcolormapObject(direction=1):
    """
    Returns a matplotlib color map object for RSA and brain plotting
    """
    if direction == 0:
        cs = ['yellow', 'red', 'gray', 'turquoise', 'blue']
    elif direction == 1:
        cs = ['blue', 'turquoise', 'gray', 'red', 'yellow']
    else:
        raise ValueError('Direction needs to be 0 or 1')
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", cs)
    return cmap


def fisher_z_transform(correlation):
    """
    Z_mean = sum[(ni - 3)Zi]/sum[ni - 3]

    :param correlation: one correlation
    :return: fisher z
    """
    z_score = 1 / 2 * np.log((1 + correlation) / (1 - correlation))

    return z_score


def cal_euclidean(pointa, pointb):
    """
    calculate Euclidean distance between point A and point B
    :param pointa:
    :param pointb:
    :return: distance
    """
    distance = np.sqrt(np.sum(np.power(pointa - pointb, 2)))

    return distance


def firstCluster(dataSets, r, include):
    cluster = []
    m = np.shape(dataSets)[0]
    ungrouped = np.array([i for i in range(m)])
    for i in range(m):
        tempCluster = []
        # 第一位存储中心点簇
        tempCluster.append(i)
        for j in range(m):
            show_progressbar('Searching center point: ', 100 * (i * m + j) / m ** 2)
            if (cal_euclidean(dataSets[i, :], dataSets[j, :]) < r and i != j):
                tempCluster.append(j)
        tempCluster = np.mat(np.array(tempCluster))
        if (np.size(tempCluster)) >= include:
            cluster.append(np.array(tempCluster).flatten())
    # 返回的是List
    center = []
    n = np.shape(cluster)[0]
    for k in range(n):
        center.append(cluster[k][0])
        # 其他的就是非中心点啦
    ungrouped = np.delete(ungrouped, center)
    # ungrouped为非中心点
    return cluster, center, ungrouped


def clusterGrouped(tempcluster, centers):
    m = np.shape(tempcluster)[0]
    group = []
    # 对应点是否遍历过
    position = np.ones(m)
    unvisited = []
    # 未遍历点
    unvisited.extend(centers)
    # 所有点均遍历完毕
    for i in range(len(position)):
        coreNeihbor = []
        result = []
        # 删除第一个
        # 刨去自己的邻居结点，这一段就类似于深度遍历
        if position[i]:
            # 将邻结点填入
            coreNeihbor.extend(list(tempcluster[i][:]))
            position[i] = 0
            temp = coreNeihbor
            # 按照深度遍历遍历完所有可达点
            # 遍历完所有的邻居结点
            while len(coreNeihbor) > 0:
                # 选择当前点
                present = coreNeihbor[0]
                for j in range(len(position)):
                    # 如果没有访问过
                    if position[j] == 1:
                        same = []
                        # 求所有的可达点
                        if (present in tempcluster[j]):
                            cluster = tempcluster[j].tolist()
                            diff = []
                            for x in cluster:
                                if x not in temp:
                                    # 确保没有重复点
                                    diff.append(x)
                            temp.extend(diff)
                            position[j] = 0
                # 删掉当前点
                del coreNeihbor[0]
                result.extend(temp)
            group.append(list(set(result)))
        i += 1
    return group


def voxel_cluster(img_data, r, include, threshold):
    """
    :param threshold: determine positive or negative value of data. e.g.-0.5 or 0.5
    :param include: min number of voxel in one including space
    :param r: radius in dbscan algorithm
    :param img_data: shape of (x, y, z), consist of 0 and non-zero number which represents significant voxel
    :return:
    """

    flat_img_data = img_data.flatten()
    if threshold > 0:
        flat_mask = (flat_img_data > threshold)
    else:
        flat_mask = (flat_img_data < threshold)
    x, y, z = img_data.shape
    data_length = x * y * z
    all_voxel_index = np.arange(data_length)
    sig_index = all_voxel_index[flat_mask]

    x_flat = np.array([tx for tz in np.arange(z) for ty in np.arange(y) for tx in np.arange(x)]).reshape(
        [data_length, 1])
    y_flat = np.array([ty for tz in np.arange(z) for ty in np.arange(y) for tx in np.arange(x)]).reshape(
        [data_length, 1])
    z_flat = np.array([tz for tz in np.arange(z) for ty in np.arange(y) for tx in np.arange(x)]).reshape(
        [data_length, 1])

    input_data = np.concatenate((x_flat[sig_index], y_flat[sig_index], z_flat[sig_index]), axis=1)

    tempcluster, center, ungrouped = firstCluster(input_data, r, include)
    group = clusterGrouped(tempcluster, center)

    sig_index_group = []
    for cluster in group:
        sig_index_group.append(sig_index[np.array(cluster)])

    centroid_list = []
    for coord in sig_index_group:
        xx = x_flat[coord]
        yy = y_flat[coord]
        zz = z_flat[coord]

        group_centroid = np.array((xx.mean(), yy.mean(), zz.mean()))
        centroid_list.append(group_centroid)

    return sig_index_group, centroid_list


def filter_cluster(img_data, k_value):

    """
    :param img_data: should be shaped of x, y, z; and its data should be bool value or 0/1
    :param k_value: threshold of voxel number
    :return: a dictionary containing all cluster mask
    """

    img_mask = np.array(img_data, dtype='bool')
    img_labels, cluster_num = ndimage.label(img_mask)

    all_cluster = {}
    for cls in np.arange(1, cluster_num + 1):
        cls_mask = (img_labels == cls)
        if np.sum(cls_mask) > k_value:
            all_cluster[cls] = cls_mask

    return all_cluster


def cluster_time_point(sig_index):
    """
    for cluster_based permutation, it cluster at least 2 adjacent time points
    :param sig_index: significant time point, type is list
           e.g. [1, 3, 7, 8, 9, 11, 12, 16, 17]
    :return: time point cluster
             e.g. [7, 8, 9, 10, 11, 12]
    """
    # new_sig_index = sig_index.copy()
    # first of all, build new sig_index
    for num_index, num in enumerate(sig_index):
        if num + 1 not in sig_index and num + 2 in sig_index:
            sig_index.insert(num_index + 1, num + 1)

    # clustering
    continue_judge = False
    all_onset_list = []
    for onset in sig_index:
        if not continue_judge:
            cls_list = [onset]
        if onset + 1 in sig_index:
            cls_list.append(onset + 1)
            continue_judge = True
        else:
            continue_judge = False
        if len(cls_list) > 1 and continue_judge is not True:
            all_onset_list.append(cls_list)

    len_list = pd.Series([len(length) for length in all_onset_list])

    counting = len_list.value_counts()
    # clustering time point based on our rule: pick top2 len list, and combine adjacent cluster(less than one
    # time point gap)
    if len(counting) == 0:
        new_max = []  # means no sig time cluster
    elif counting.loc[max(len_list)] <= 2:
        # have more than 2 sig time cluster and less than 3
        if len(len_list) == 1:
            combine_cls = all_onset_list[np.argmax(len_list)]  # means only one time cluster
        else:
            # more than 2 cluster and we pick top 2 length
            # get the max cluster
            top1_cls = all_onset_list[np.argmax(len_list)]

            # then remove the max cluster and get the top2 max cluster
            all_onset_store = all_onset_list.copy()
            all_onset_store.pop(int(np.argmax(len_list)))
            len_list2 = pd.Series([len(length) for length in all_onset_store])
            top2_cls = all_onset_store[np.argmax(len_list2)]

            counting2 = len_list2.value_counts()

            if counting2.loc[max(len_list2)] == 1:
                if abs(top2_cls[0] - top1_cls[-1]) <= 2 or abs(top2_cls[-1] - top1_cls[0]) <= 2:
                    if top2_cls[0] < top1_cls[0]:
                        combine_cls = np.arange(top2_cls[0], top1_cls[-1] + 1)
                    else:
                        combine_cls = np.arange(top1_cls[0], top2_cls[-1] + 1)
                else:
                    combine_cls = np.array(top1_cls)
                # combine two max cluster if they are closed
            else:
                no_2_max_length = max(len_list2)
                all_no2max = [max_cls2 for max_cls2 in all_onset_list if len(max_cls2) == no_2_max_length]

                for no2_cls in all_no2max[::-1]:
                    if abs(no2_cls[0] - top1_cls[-1]) <= 2 or abs(no2_cls[-1] - top1_cls[0]) <= 2:
                        if no2_cls[0] < top1_cls[0]:
                            combine_cls = np.arange(no2_cls[0], top1_cls[-1] + 1)
                            break
                        else:
                            combine_cls = np.arange(top1_cls[0], no2_cls[-1] + 1)
                            break
                    else:
                        combine_cls = np.array(top1_cls)

        new_max = combine_cls

    else:
        # appear more than 3 same length list, and we combine rear element preferentially
        max_length = max(len_list)

        all_max = [max_cls for max_cls in all_onset_list if len(max_cls) == max_length]

        new_max = all_max.copy()
        m_index = -1
        while len(new_max) != 1:
            now_max = new_max[m_index]
            next_max = new_max[m_index - 1]
            if now_max[0] - next_max[-1] > 2:
                if len(next_max) == len(now_max):
                    new_max.pop(m_index)
                else:
                    new_max.pop(m_index - 1)
            else:
                combine_max = list(np.arange(next_max[0], now_max[-1] + 1))
                new_max.pop(m_index - 1)
                new_max.pop(m_index)
                new_max.append(combine_max)
        new_max = new_max[0]

    return list(new_max)


def ersp_baseline_normalize(epochs, baseline, base, sample_rate):

    """
    this function apply for performing ersp baseline normalization based on mne package

    :param epochs: epochs object
    :param baseline: e.g. [-500, -200]
    :param base: e.g. trial onset: -1000ms
    :param sample_rate: e.g. 1000Hz
    :return:
    """

    epochs_copy = epochs.copy()

    for trial in np.arange(len(epochs_copy)):
        for chan in np.arange(len(epochs_copy.ch_names)):
            epochs_copy._data[trial, chan] = 10 * np.log10(
                epochs_copy._data[trial, chan] / np.average(
                    epochs_copy._data[trial, chan,
                    int((baseline[0] - base)/1000 * sample_rate): int((baseline[1] - base)/1000 * sample_rate)]))

    return epochs_copy


def save_variable(variable, filename):
    """
    save the object by pickle site-package
    :param variable: object you want to save
    :param filename: saving path
    :return:
    """
    f = open(filename, 'wb')
    pickle.dump(variable, f)
    f.close()
    return filename


def load_variable(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


' a function for time-by-time decoding for EEG-like data (cross validation) '


def temporal_decoding(data, labels, n=2, navg=5, time_opt="average", time_win=5, time_step=5, nfolds=5, nrepeats=2,
                      normalization=False, pca=False, pca_components=0.95, smooth=True, kernel_function='linear',
                      c_value=1):

    """
    Conduct time-by-time decoding for EEG-like data (cross validation)

    Parameters
    ----------
    data : array
        The neural data.
        The shape of data must be [n_subs, n_trials, n_chls, n_ts]. n_subs, n_trials, n_chls and n_ts represent the
        number of subjects, the number of trails, the number of channels and the number of time-points.
    labels : array
        The labels of each trial.
        The shape of labels must be [n_subs, n_trials]. n_subs and n_trials represent the number of subjects and the
        number of trials.
    n : int. Default is 2.
        The number of categories for classification.
    navg : int. Default is 5.
        The number of trials used to average.
    time_opt : string "average" or "features". Default is "average".
        Average the time-points or regard the time points as features for classification
        If time_opt="average", the time-points in a certain time-window will be averaged.
        If time_opt="features", the time-points in a certain time-window will be used as features for classification.
    time_win : int. Default is 5.
        Set a time-window for decoding for different time-points.
        If time_win=5, that means each decoding process based on 5 time-points.
    time_step : int. Default is 5.
        The time step size for each time of decoding.
    nfolds : int. Default is 5.
        The number of folds.
        k should be at least 2.
    nrepeats : int. Default is 2.
        The times for iteration.
    normalization : boolean True or False. Default is False.
        Normalize the data or not.
    pca : boolean True or False. Default is False.
        Apply principal component analysis (PCA).
    pca_components : int or float. Default is 0.95.
        Number of components for PCA to keep. If 0<pca_components<1, select the numbder of components such that the
        amount of variance that needs to be explained is greater than the percentage specified by pca_components.
    smooth : boolean True or False, or int. Default is True.
        Smooth the decoding result or not.
        If smooth = True, the default smoothing step is 5. If smooth = n (type of n: int), the smoothing step is n.
    kernel_function: str list ----- {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}. Default is linear.
        Select the kernel function applied for SVM.
    c_value: float. Default is 1.0.
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive. The penalty
        is a squared l2 penalty.

    Returns
    -------
    accuracies : array
        The time-by-time decoding accuracies.
        The shape of accuracies is [n_subs, int((n_ts-time_win)/time_step)+1].
    """

    if np.shape(data)[0] != np.shape(labels)[0]:

        print("\nThe number of subjects of data doesn't match the number of subjects of labels.\n")

        return "Invalid input!"

    if np.shape(data)[1] != np.shape(labels)[1]:

        print("\nThe number of epochs doesn't match the number of labels.\n")

        return "Invalid input!"

    nsubs, ntrials, nchls, nts = np.shape(data)

    ncategories = np.zeros([nsubs], dtype=int)

    labels = np.array(labels)

    for sub in range(nsubs):

        sublabels_set = set(labels[sub].tolist())
        ncategories[sub] = len(sublabels_set)

    if len(set(ncategories.tolist())) != 1:

        print("\nInvalid labels!\n")

        return "Invalid input!"

    if n != ncategories[0]:

        print("\nThe number of categories for decoding doesn't match ncategories (" + str(ncategories) + ")!\n")

        return "Invalid input!"

    categories = list(sublabels_set)

    newnts = int((nts-time_win)/time_step)+1

    if time_opt == "average":

        avgt_data = np.zeros([nsubs, ntrials, nchls, newnts])

        for t in range(newnts):
            avgt_data[:, :, :, t] = np.average(data[:, :, :, t * time_step:t * time_step + time_win], axis=3)

        acc = np.zeros([nsubs, newnts])

        total = nsubs * nrepeats * newnts * nfolds

        for sub in range(nsubs):

            ns = np.zeros([n], dtype=int)

            for i in range(ntrials):
                for j in range(n):
                    if labels[sub, i] == categories[j]:
                        ns[j] = ns[j] + 1

            minn = int(np.min(ns) / navg)

            subacc = np.zeros([nrepeats, newnts, nfolds])

            for i in range(nrepeats):

                datai = np.zeros([n, minn * navg, nchls, newnts])
                labelsi = np.zeros([n, minn], dtype=int)

                for j in range(n):
                    labelsi[j] = j

                randomindex = np.random.permutation(np.array(range(ntrials)))

                m = np.zeros([n], dtype=int)

                for j in range(ntrials):
                    for k in range(n):

                        if labels[sub, randomindex[j]] == categories[k] and m[k] < minn * navg:
                            datai[k, m[k]] = avgt_data[sub, randomindex[j]]
                            m[k] = m[k] + 1

                avg_datai = np.zeros([n, minn, nchls, newnts])

                for j in range(minn):
                    avg_datai[:, j] = np.average(datai[:, j * navg:j * navg + navg], axis=1)

                x = np.reshape(avg_datai, [n * minn, nchls, newnts])
                y = np.reshape(labelsi, [n * minn])

                for t in range(newnts):

                    state = np.random.randint(0, 100)
                    kf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=state)
                    xt = x[:, :, t]

                    fold_index = 0
                    for train_index, test_index in kf.split(xt, y):

                        x_train = xt[train_index]
                        x_test = xt[test_index]

                        if normalization is True:
                            scaler = StandardScaler()
                            x_train = scaler.fit_transform(x_train)
                            x_test = scaler.transform(x_test)

                        if pca is True:

                            Pca = PCA(n_components=pca_components)
                            x_train = Pca.fit_transform(x_train)
                            x_test = Pca.transform(x_test)

                        svm = SVC(kernel=kernel_function, tol=1e-4, C=c_value, probability=False)
                        svm.fit(x_train, y[train_index])
                        subacc[i, t, fold_index] = svm.score(x_test, y[test_index])

                        percent = (sub * nrepeats * newnts * nfolds + i * newnts * nfolds + t * nfolds + fold_index + 1) / total * 100
                        show_progressbar("Calculating", percent)

                        if sub == (nsubs - 1) and i == (nrepeats - 1) and t == (newnts - 1) and fold_index == (
                                nfolds - 1):
                            print("\nDecoding finished!\n")

                        fold_index = fold_index + 1

            acc[sub] = np.average(subacc, axis=(0, 2))

    if time_opt == "features":

        avgt_data = np.zeros([nsubs, ntrials, nchls, time_win, newnts])

        for t in range(newnts):
            avgt_data[:, :, :, :, t] = data[:, :, :, t * time_step:t * time_step + time_win]

        avgt_data = np.reshape(avgt_data, [nsubs, ntrials, nchls*time_win, newnts])

        acc = np.zeros([nsubs, newnts])

        total = nsubs * nrepeats * newnts * nfolds

        for sub in range(nsubs):

            ns = np.zeros([n], dtype=int)

            for i in range(ntrials):
                for j in range(n):
                    if labels[sub, i] == categories[j]:
                        ns[j] = ns[j] + 1

            minn = int(np.min(ns) / navg)

            subacc = np.zeros([nrepeats, newnts, nfolds])

            for i in range(nrepeats):

                datai = np.zeros([n, minn * navg, nchls * time_win, newnts])
                labelsi = np.zeros([n, minn], dtype=int)

                for j in range(n):
                    labelsi[j] = j

                randomindex = np.random.permutation(np.array(range(ntrials)))

                m = np.zeros([n], dtype=int)

                for j in range(ntrials):
                    for k in range(n):

                        if labels[sub, randomindex[j]] == categories[k] and m[k] < minn * navg:
                            datai[k, m[k]] = avgt_data[sub, randomindex[j]]
                            m[k] = m[k] + 1

                avg_datai = np.zeros([n, minn, nchls * time_win, newnts])

                for j in range(minn):
                    avg_datai[:, j] = np.average(datai[:, j * navg:j * navg + navg], axis=1)

                x = np.reshape(avg_datai, [n * minn, nchls * time_win, newnts])
                y = np.reshape(labelsi, [n * minn])

                for t in range(newnts):

                    state = np.random.randint(0, 100)
                    kf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=state)
                    xt = x[:, :, t]

                    fold_index = 0
                    for train_index, test_index in kf.split(xt, y):

                        x_train = xt[train_index]
                        x_test = xt[test_index]

                        if normalization is True:
                            scaler = StandardScaler()
                            x_train = scaler.fit_transform(x_train)
                            x_test = scaler.transform(x_test)

                        if pca is True:

                            Pca = PCA(n_components=pca_components)
                            x_train = Pca.fit_transform(x_train)
                            x_test = Pca.transform(x_test)

                        svm = SVC(kernel='linear', tol=1e-4, probability=False)
                        svm.fit(x_train, y[train_index])
                        subacc[i, t, fold_index] = svm.score(x_test, y[test_index])

                        percent = (sub * nrepeats * newnts * nfolds + i * newnts * nfolds + t * nfolds + fold_index + 1) / total * 100
                        show_progressbar("Calculating", percent)

                        if sub == (nsubs - 1) and i == (nrepeats - 1) and t == (newnts - 1) and fold_index == (
                                nfolds - 1):
                            print("\nDecoding finished!\n")

                        fold_index = fold_index + 1

            acc[sub] = np.average(subacc, axis=(0, 2))

    if smooth is False:

        return acc

    if smooth is True:

        smooth_acc = smooth_1d(acc)

        return smooth_acc

    else:

        smooth_acc = smooth_1d(acc, n=smooth)

        return smooth_acc


def cross_temporal_decoding(data, labels, test_data, test_labels, n=2, navg=5, time_win=5,
                            time_step=5, nrepeats=2,
                            normalization=False, smooth=True):

    """
    Conduct cross-temporal decoding for EEG-like data (cross validation)

    Parameters
    ----------
    data : array
        The neural data.
        The shape of data must be [n_subs, n_trials, n_chls, n_ts]. n_subs, n_trials, n_chls and n_ts represent the
        number of subjects, the number of trails, the number of channels and the number of time-points.
    labels : array
        The labels of each trial.
        The shape of labels must be [n_subs, n_trials]. n_subs and n_trials represent the number of subjects and the
        number of trials.
    test_data: array
        The neural testing data.
        The shape of data must be [n_subs, n_trials, n_chls, n_ts]. n_subs, n_trials, n_chls and n_ts represent the
        number of subjects, the number of trails, the number of channels and the number of time-points.
    test_labels: array
        he labels of each testing trial.
        The shape of labels must be [n_subs, n_trials]. n_subs and n_trials represent the number of subjects and the
        number of trials.
    n : int. Default is 2.
        The number of categories for classification.
    navg : int. Default is 5.
        The number of trials used to average.
    time_win : int. Default is 5.
        Set a time-window for decoding for different time-points.
        If time_win=5, that means each decoding process based on 5 time-points.
    time_step : int. Default is 5.
        The time step size for each time of decoding.
    nrepeats : int. Default is 2.
        The times for iteration.
    normalization : boolean True or False. Default is False.
        Normalize the data or not.
    smooth : boolean True or False, or int. Default is True.
        Smooth the decoding result or not.
        If smooth = True, the default smoothing step is 5. If smooth = n (type of n: int), the smoothing step is n.

    Returns
    -------
    accuracies : array
        The cross-temporal decoding accuracies.
        The shape of accuracies is [n_subs, int((n_ts-time_win)/time_step)+1, int((n_ts-time_win)/time_step)+1].
    """

    if np.shape(data)[0] != np.shape(labels)[0]:

        print("\nThe number of subjects of data doesn't match the number of subjects of labels.\n")

        return "Invalid input!"

    if np.shape(data)[1] != np.shape(labels)[1]:

        print("\nThe number of epochs doesn't match the number of labels.\n")

        return "Invalid input!"

    nsubs, ntrials, nchls, nts = np.shape(data)

    test_n_sub, test_n_trials, test_n_chls, test_n_ts = np.shape(test_data)

    ncategories = np.zeros([nsubs], dtype=int)

    labels = np.array(labels)
    test_labels = np.array(test_labels)

    for sub in range(nsubs):
        sublabels_set = set(labels[sub].tolist())
        ncategories[sub] = len(sublabels_set)

    if nchls != test_n_chls:
        print("\nInvalid labels!\n")
        return "Invalid input!"

    if nts != test_n_ts:
        print("\nInvalid labels!\n")
        return "Invalid input!"

    if len(set(ncategories.tolist())) != 1:

        print("\nInvalid labels!\n")

        return "Invalid input!"

    if n != ncategories[0]:

        print("\nThe number of categories for decoding doesn't match ncategories (" + str(ncategories) + ")!\n")

        return "Invalid input!"

    categories = list(sublabels_set)

    newnts = int((nts-time_win)/time_step)+1

    # if time_opt == 'feature'

    avgt_data = np.zeros([nsubs, ntrials, nchls, newnts])
    avgt_test_data = np.zeros([test_n_sub, test_n_trials, test_n_chls, newnts])

    for t in range(newnts):
        avgt_data[:, :, :, t] = np.average(data[:, :, :, t * time_step:t * time_step + time_win], axis=3)
        avgt_test_data[:, :, :, t] = np.average(test_data[:, :, :, t * time_step:t * time_step + time_win], axis=3)

    acc = np.zeros([nsubs, newnts, newnts])

    total = nsubs * nrepeats * newnts

    print("\nDecoding")

    for sub in range(nsubs):

        ns = np.zeros([n], dtype=int)
        for i in range(ntrials):
            for j in range(n):
                if labels[sub, i] == categories[j]:
                    ns[j] = ns[j] + 1

        test_ns = np.zeros([n], dtype=int)
        for i in range(test_n_trials):
            for j in range(n):
                if test_labels[sub, i] == categories[j]:
                    test_ns[j] = test_ns[j] + 1

        minn = int(np.min(ns) / navg)
        test_minn = int(np.min(test_ns) / navg)

        subacc = np.zeros([nrepeats, newnts, newnts])

        for i in range(nrepeats):

            datai = np.zeros([n, minn * navg, nchls, newnts])
            labelsi = np.zeros([n, minn], dtype=int)

            test_datai = np.zeros([n, test_minn * navg, test_n_chls, newnts])
            test_labeli = np.zeros([n, test_minn], dtype=int)

            for j in range(n):
                labelsi[j] = j
                test_labeli[j] = j

            randomindex = np.random.permutation(np.array(range(ntrials)))

            m = np.zeros([n], dtype=int)

            for j in range(ntrials):
                for k in range(n):
                    if labels[sub, randomindex[j]] == categories[k] and m[k] < minn * navg:
                        datai[k, m[k]] = avgt_data[sub, randomindex[j]]
                        m[k] = m[k] + 1

            test_randomindex = np.random.permutation(np.array(range(test_n_trials)))

            test_m = np.zeros([n], dtype=int)

            for j in range(test_n_trials):
                for k in range(n):
                    if test_labels[sub, test_randomindex[j]] == categories[k] and test_m[k] < test_minn * navg:
                        test_datai[k, test_m[k]] = avgt_test_data[sub, test_randomindex[j]]
                        test_m[k] = test_m[k] + 1

            avg_datai = np.zeros([n, minn, nchls, newnts])
            test_avg_datai = np.zeros([n, test_minn, test_n_chls, newnts])

            for j in range(minn):
                avg_datai[:, j] = np.average(datai[:, j * navg:j * navg + navg], axis=1)

            for j in range(test_minn):
                test_avg_datai[:, j] = np.average(test_datai[:, j * navg:j * navg + navg], axis=1)


            x = np.reshape(avg_datai, [n * minn, nchls, newnts])
            y = np.reshape(labelsi, [n * minn])

            test_x = np.reshape(test_avg_datai, [n * test_minn, test_n_chls, newnts])
            test_y = np.reshape(test_labeli, [n * test_minn])

            for t in range(newnts):
                xt = x[:, :, t]
                test_xt = test_x[:, :, t]

                percent = (sub * nrepeats * newnts + i * newnts + t + 1) / total * 100
                show_progressbar("Calculating", percent)

                if normalization is True:

                    scaler = StandardScaler()
                    x_train = scaler.fit_transform(xt)
                    svm = SVC(kernel='linear', tol=1e-4, probability=False)
                    svm.fit(x_train, y)
                    subacc[i, t, t] = svm.score(scaler.transform(test_xt), test_y)

                    for tt in range(newnts - 1):
                        if tt < t:
                            test_xtt = test_x[:, :, tt]
                            subacc[i, t, tt] = svm.score(scaler.transform(test_xtt), test_y)
                        if tt >= t:
                            test_xtt = test_x[:, :, tt + 1]
                            subacc[i, t, tt + 1] = svm.score(scaler.transform(test_xtt), test_y)

                if normalization is False:
                    svm = SVC(kernel='linear', tol=1e-4, probability=False)
                    svm.fit(xt, y)
                    subacc[i, t, t] = svm.score(test_xt, test_y)

                    for tt in range(newnts - 1):
                        if tt < t:
                            test_xtt = test_x[:, :, tt]
                            subacc[i, t, tt] = svm.score(test_xtt, test_y)
                        if tt >= t:
                            test_xtt = test_x[:, :, tt + 1]
                            subacc[i, t, tt + 1] = svm.score(test_xtt, test_y)

                if sub == (nsubs - 1) and i == (nrepeats - 1) and t == (newnts - 1):
                    print("\nDecoding finished!\n")

        acc[sub] = np.average(subacc, axis=0)

    if smooth is False:
        return acc

    if smooth is True:

        smooth_acc = smooth_2d(acc)

        return smooth_acc

    else:

        smooth_acc = smooth_2d(acc, n=smooth)

        return smooth_acc


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

    event_df['dp_stim'] = np.array(ges_index)
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


def plot_2d_sig_results(stats_results, x_axis, y_axis):
    padsats_results = np.zeros([len(x_axis) + 2, len(y_axis) + 2])
    padsats_results[1:len(x_axis) + 1, 1:len(y_axis) + 1] = stats_results

    time_x = np.concatenate(([x_axis[0] - 1], x_axis, [x_axis[-1] + 1]))
    time_y = np.concatenate(([y_axis[0] - 1], y_axis, [y_axis[-1] + 1]))
    time_X, time_Y = np.meshgrid(time_x, time_y)
    plt.contour(time_X, time_Y, padsats_results, [0.5], colors="blue", alpha=0.9,
                linewidths=2, linestyles="dashed")
    plt.contour(time_X, time_Y, padsats_results, [-0.5], colors="red", alpha=0.9,
                linewidths=2, linestyles="dashed")

def kde_entropy(x, bandwidth=0.5):
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(x.reshape(-1, 1))
    log_dens = kde.score_samples(x.reshape(-1, 1))
    return -np.mean(log_dens)

def continuous_mutual_information(x, y, bandwidth=0.5):
    h_x = kde_entropy(x, bandwidth)
    h_y = kde_entropy(y, bandwidth)

    xy = np.concatenate((x, y))
    h_xy = kde_entropy(xy, bandwidth)

    mi = h_x + h_y - h_xy
    return mi


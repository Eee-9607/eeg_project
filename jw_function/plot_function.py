# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal
from scipy.stats import ttest_1samp, ttest_rel
from nilearn import plotting, datasets, surface
import nibabel as nib
from neurora.stuff import get_affine, get_bg_ch2, get_bg_ch2bet, correct_by_threshold, \
    clusterbased_permutation_1d_1samp_1sided, clusterbased_permutation_2d_1samp_1sided, \
    clusterbased_permutation_1d_1samp_2sided, clusterbased_permutation_2d_2sided, smooth_1d
from decimal import Decimal
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
import statsmodels.api as sm
from statsmodels.stats.multitest import fdrcorrection
import pingouin as pg
from scipy import stats
import seaborn as sns
import sys
import matplotlib.colors as colors
from skimage.measure import label
import matplotlib.tri as tri
from matplotlib.colors import LinearSegmentedColormap



# adjust parameter from neuroRA plotting function
def plot_temporal_decoding_acc(acc, start_time=0, end_time=1, time_interval=0.01, chance=0.5, p=0.05, cbpt=True,
                               clusterp=0.05, stats_time=[0, 1], color='r', xlim=[0, 1], ylim=[0.4, 0.8],
                               xlabel='Time (s)', ylabel='Decoding Accuracy', figsize=[6.4, 3.6], x0=0, ticksize=12,
                               fontsize=16, markersize=2, title=None, title_fontsize=16, avgshow=False,
                               save_path=False):

    """
    Plot the time-by-time decoding accuracies

    Parameters
    ----------
    acc : array
        The decoding accuracies.
        The size of acc should be [n_subs, n_ts]. n_subs, n_ts represent the number of subjects and number of
        time-points.
    start_time : int or float. Default is 0.
        The start time.
    end_time : int or float. Default is 1.
        The end time.
    time_interval : float. Default is 0.01.
        The time interval between two time samples.
    chance : float. Default is 0.5.
        The chance level.
    p : float. Default is 0.05.
        The threshold of p-values.
    cbpt : bool True or False. Default is True.
        Conduct cluster-based permutation test or not.
    clusterp : float. Default is 0.05.
        The threshold of cluster-defining p-values.
    stats_time : array or list [stats_time1, stats_time2]. Default os [0, 1].
        Time period for statistical analysis.
    color : matplotlib color or None. Default is 'r'.
        The color for the curve.
    xlim : array or list [xmin, xmax]. Default is [0, 1].
        The x-axis (time) view lims.
    ylim : array or list [ymin, ymax]. Default is [0.4, 0.8].
        The y-axis (decoding accuracy) view lims.
    xlabel : string. Default is 'Time (s)'.
        The label of x-axis.
    ylabel : string. Default is 'Decoding Accuracy'.
        The label of y-axis.
    figsize : array or list, [size_X, size_Y]. Default is [6.4, 3.6].
        The size of the figure.
    x0 : float. Default is 0.
        The Y-axis is at x=x0.
    ticksize : int or float. Default is 12.
        The size of the ticks.
    fontsize : int or float. Default is 16.
        The fontsize of the labels.
    markersize : int or float. Default is 2.
        The size of significant marker.
    title : string-array. Default is None.
        The title of the figure.
    title_fontsize : int or float. Default is 16.
        The fontsize of the title.
    avgshow : boolen True or False. Default is False.
        Show the averaging decoding accuracies or not.
    save_path: string. Default is False.
        Save directory of the figure
    """

    if len(np.shape(acc)) != 2:

        return "Invalid input!"

    nsubs, nts = np.shape(acc)
    tstep = float(Decimal((end_time - start_time) / nts).quantize(Decimal(str(time_interval))))

    if tstep != time_interval:

        return "Invalid input!"

    delta1 = (stats_time[0] - start_time) / tstep - int((stats_time[0] - start_time) / tstep)
    delta2 = (stats_time[1] - start_time) / tstep - int((stats_time[1] - start_time) / tstep)
    if delta1 == 0:
        stats_time1 = int((stats_time[0] - start_time) / tstep)
    else:
        stats_time1 = int((stats_time[0] - start_time) / tstep) + 1
    if delta2 == 0:
        stats_time2 = int((stats_time[1] - start_time) / tstep)
    else:
        stats_time2 = int((stats_time[1] - start_time) / tstep) + 1

    yminlim = ylim[0]
    ymaxlim = ylim[1]

    avg = np.average(acc, axis=0)
    err = np.zeros([nts])
    for t in range(nts):
        err[t] = np.std(acc[:, t], ddof=1) / np.sqrt(nsubs)

    if cbpt == True:

        ps_stats = clusterbased_permutation_1d_1samp_1sided(acc[:, stats_time1:stats_time2], level=chance,
                                                            p_threshold=p, clusterp_threshold=clusterp, iter=1000)
        ps = np.zeros([nts])
        ps[stats_time1:stats_time2] = ps_stats

    else:
        ps = np.zeros([nts])
        for t in range(nts):
            if t >= stats_time1 and t< stats_time2:
                ps[t] = ttest_1samp(acc[:, t], chance, alternative="greater")[1]
                if ps[t] < p:
                    ps[t] = 1
                else:
                    ps[t] = 0

    print('\nSignificant time-windows:')
    for t in range(nts):
        if t == 0 and ps[t] == 1:
            print(str(int(start_time * 1000)) + 'ms to ', end='')
        if t > 0 and ps[t] == 1 and ps[t - 1] == 0:
            print(str(int((start_time + t * tstep) * 1000)) + 'ms to ', end='')
        if t < nts - 1 and ps[t] == 1 and ps[t + 1] == 0:
            print(str(int((start_time + (t + 1) * tstep) * 1000)) + 'ms')
        if t == nts - 1 and ps[t] == 1:
            print(str(int(stats_time[1] * 1000)) + 'ms')

    for t in range(nts):
        if ps[t] == 1:
            plt.plot(t*tstep+start_time+0.5*tstep, (ymaxlim-yminlim)*0.95+yminlim, 's', color=color, alpha=0.8,
                     markersize=markersize)
            xi = [t*tstep+start_time, t*tstep+tstep+start_time]
            ymin = [chance]
            ymax = [avg[t] - err[t]]
            plt.fill_between(xi, ymax, ymin, facecolor=color, alpha=0.2)

    fig = plt.gcf()
    fig.set_size_inches(figsize[0], figsize[1])
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(3)
    ax.spines["left"].set_position(("data", x0))
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["bottom"].set_position(("data", chance))
    x = np.arange(start_time+0.5*tstep, end_time+0.5*tstep, tstep)
    if avgshow is True:
        plt.plot(x, avg, color=color, alpha=0.95)
    plt.fill_between(x, avg+err, avg-err, facecolor=color, alpha=0.75)
    plt.ylim(yminlim, ymaxlim)
    plt.xlim(xlim[0], xlim[1])
    plt.tick_params(labelsize=ticksize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)

    plt.title(title, fontsize=title_fontsize)
    if save_path:
        plt.savefig(save_path)
        plt.show()

    return ps


def star_sig(p):
    """
    change p value to star string
    Args:
        p: p value

    Returns:
        star string

    """
    if p < .001:
        return '***'
    elif p < .01:
        return '**'
    elif p < .05:
        return '*'
    else:
        return 'ns'


def plot_bar_with_stats(data, x, y, hue, x_label, hue_label, save_path, title,
                        y_lim=None, hue_stats_opt=True, dpi=600):


    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=x, y=y, hue=hue, data=data, palette="deep", errorbar='se')
    ax.set(xlabel=x, ylabel=y, title=title)

    if y_lim:
        ax.set_ylim(y_lim)

    ax.set_xticklabels(x_label, rotation=45)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=hue_label)

    if hue_stats_opt:
        x_label_num = np.unique(data[x])
        hue_label_num = np.unique(data[hue])
        level_num = len(hue_label)
        if level_num == 2:
            for x_index in x_label_num:
                x_data = data.loc[data.loc[:, x] == x_index, :]
                first_hue = x_data.loc[x_data.loc[:, hue] == hue_label_num[0], :]
                second_hue = x_data.loc[x_data.loc[:, hue] == hue_label_num[1], :]

                all_hue = (x_data.loc[x_data.loc[:, hue] == hue_label_num[0], :],
                           x_data.loc[x_data.loc[:, hue] == hue_label_num[1], :])
                all_data = (first_hue[y].mean(), second_hue[y].mean())

                max_data = np.max(all_data) + 1.3 * all_hue[np.argmax(all_data)]['rt'].sem()

                t_stats = pg.ttest(first_hue[y], second_hue[y])

                star_str = star_sig(t_stats.loc['T-test', 'p-val'])

                ax.text(x_index, max_data, star_str, color='black', ha='center')

    fig.tight_layout()
    fig.subplots_adjust(top=0.8, bottom=0.2, left=0.2, right=0.9)
    fig.savefig(save_path, dpi=dpi)

    return fig


def plot_violin_with_stats(data, x, y, hue, x_label, hue_label, save_path, title,
                           y_lim=None, hue_stats_opt=True, dpi=600):

    sns.set(style="whitegrid")
    fig = plt.figure(figsize=(10, 6))
    ax = sns.violinplot(x=x, y=y, hue=hue, data=data, palette="deep", errorbar='se')
    ax.set(xlabel=x, ylabel=y, title=title)

    if y_lim:
        ax.set_ylim(y_lim)

    ax.set_xticklabels(x_label, rotation=45)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=hue_label)

    if hue_stats_opt:
        x_label_num = np.unique(data[x])
        hue_label_num = np.unique(data[hue])
        level_num = len(hue_label)
        if level_num == 2:
            for x_index in x_label_num:
                x_data = data.loc[data.loc[:, x] == x_index, :]
                first_hue = x_data.loc[x_data.loc[:, hue] == hue_label_num[0], :]
                second_hue = x_data.loc[x_data.loc[:, hue] == hue_label_num[1], :]

                all_hue = (x_data.loc[x_data.loc[:, hue] == hue_label_num[0], :],
                           x_data.loc[x_data.loc[:, hue] == hue_label_num[1], :])
                all_data = (first_hue[y].mean(), second_hue[y].mean())

                max_data = np.max(all_data) + 1.3 * all_hue[np.argmax(all_data)]['rt'].sem()

                t_stats = pg.ttest(first_hue[y], second_hue[y])

                star_str = star_sig(t_stats.loc['T-test', 'p-val'])

                ax.text(x_index, max_data, star_str, color='black', ha='center')

    fig.tight_layout()
    fig.subplots_adjust(top=0.8, bottom=0.2, left=0.2, right=0.9)
    fig.savefig(save_path, dpi=dpi)

    return fig


def show_progressbar(string, cur, total=100):
    """
    just for checking progress, copy code in neuroRA's rsa_plot
    :param string: word you want to present in front of progress
    :param cur: the current value
    :param total: total value
    :return:
    """
    percent = '{:.2%}'.format(cur / total)
    sys.stdout.write('\r')
    sys.stdout.write(string + ": [%-100s] %s" % ('=' * int(cur), percent))
    sys.stdout.flush()


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):

    """
    for plotting some gradient color
    Args:
        cmap: colormap
        minval: minium value in the colormap
        maxval: maxium value in the colormap
        n: bin number

    Returns:

    """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n}, {a:.2f}, {b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap


def plotting_sig_ts_cluster(time_series, real_value, stats_ts, ax, line_color, line_label, shadow_color, min_value):
    ax.plot(time_series, real_value, color=line_color,
            label=line_label)

    permu_labels = label(stats_ts, connectivity=1)
    permu_nclusters = int(np.max(permu_labels))

    for k in np.arange(1, permu_nclusters + 1):
        sig_time = (permu_labels == k)
        hline = np.array([min_value for i in np.arange(np.sum(sig_time))])

        ax.fill_between(time_series[sig_time], hline,
                        real_value[sig_time], alpha=0.5,
                        facecolor=shadow_color)


def group_contrast_figure(data, x, y, hue=None, x_label=None, hue_label=None, save_path=None, title=None,
                          color_list=None, y_lim=None, hue_stats_opt=True, dpi=600):

    # sns.set(style="whitegrid")
    fig = plt.figure(figsize=(10, 6))

    sns.violinplot(data=data, x=x, y=y, palette=color_list)

    # 画出雨云图（半个小提琴图后面加上散点图和箱型图）
    sns.boxplot(x='condition', y='value', data=df, whis=np.inf, fliersize=0, width=0.2, color='lightgray', ax=ax)
    sns.stripplot(x='condition', y='value', data=df, color='black', alpha=0.5, size=4, jitter=True, ax=ax)

    # 用循环绘制每个条件
    positions = {'A': 0, 'B': 1, 'C': 2}
    for condition in ['A', 'B', 'C']:
        half_violinplot(data=df[df['condition'] == condition]['value'], ax=ax, position=positions[condition],
                        color='lightblue', palette='pastel')

        # 标题和标签
    plt.title('Half Violin Plot with Raincloud Effect')
    plt.xlabel('Condition')
    plt.ylabel('Value')
    plt.show()

    ax = sns.violinplot(x=x, y=y, hue=hue, data=color_list, palette="deep", errorbar='se')
    ax.set(xlabel=x, ylabel=y, title=title)

    if y_lim:
        ax.set_ylim(y_lim)

    ax.set_xticklabels(x_label, rotation=45)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=hue_label)

    if hue_stats_opt:
        x_label_num = np.unique(data[x])
        hue_label_num = np.unique(data[hue])
        level_num = len(hue_label)
        if level_num == 2:
            for x_index in x_label_num:
                x_data = data.loc[data.loc[:, x] == x_index, :]
                first_hue = x_data.loc[x_data.loc[:, hue] == hue_label_num[0], :]
                second_hue = x_data.loc[x_data.loc[:, hue] == hue_label_num[1], :]

                all_hue = (x_data.loc[x_data.loc[:, hue] == hue_label_num[0], :],
                           x_data.loc[x_data.loc[:, hue] == hue_label_num[1], :])
                all_data = (first_hue[y].mean(), second_hue[y].mean())

                max_data = np.max(all_data) + 1.3 * all_hue[np.argmax(all_data)]['rt'].sem()

                t_stats = pg.ttest(first_hue[y], second_hue[y])

                star_str = star_sig(t_stats.loc['T-test', 'p-val'])

                ax.text(x_index, max_data, star_str, color='black', ha='center')

    fig.tight_layout()
    fig.subplots_adjust(top=0.8, bottom=0.2, left=0.2, right=0.9)

    if save_path:
        fig.savefig(save_path, dpi=dpi)

    return fig

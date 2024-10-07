# -*- coding = utf-8 -*-
# @Time : 2023/10/19 上午11:18
# @Author : yangjw
# @Site : 
# @File : rsa_function.py
# @Software : PyCharm

import warnings
import numpy as np
from statsmodels.api import OLS
from scipy.stats import t


def rsa_glm(eeg_rdm, bhv_rdm, normalization=True, warn_opt=True):

    """
    Args:
        eeg_rdm: array. The size of eeg_rdm should be [time_point, sub_num, n].
                            time_point is the number of sample point of the timeseries in eeg/meg case.
                        If it's fMRI data, you should set time_point is 1, then calculate rdm weight of single
                        roi/vertex/voxel one by one, or input a flattened data, then reshape time_point dimension of
                        outcome to [x, y, z].
                            sub_num is the number of subject.
                            n is number of element in single matrix, which is meaning that you should flatten your rdm
        bhv_rdm: array. The size of bhv_rdm should be [m, sub_num, n].
                        m is number of behavior matrix/theory matrix

        normalization: bool object.If true, normalize RDM for following calculation.Default is True.

    Returns:
        beta_t_array: T values of every beta of single matrix. shape: [m, time_point, sub_num]
        t_array: T values of beta difference. shape: [C(m, 2), time_point, sub_num].
                 C(m, 2) represents Combination of choosing 2 items from m items.
        index_list: Indices combination of behavior matrices
        r_square: R-squared
        r_square_adj: adjusted R-squared
        bic: Bayesian information criterion
        aic: Akaike information criterion

    """

    time_point, sub_num, n = eeg_rdm.shape
    m = bhv_rdm.shape[0]
    df = n - m - 1

    beta_list = np.arange(m)

    index_list = []  # generate indices of 2 bhv matrix

    for i in beta_list:
        left_index = beta_list[(beta_list != i) & (beta_list > i)]
        for k in left_index:
            index_list.append((i, k))

    t_array = np.zeros([len(index_list), time_point, sub_num])  # save t value of beta difference
    beta_t_array = np.zeros([m, time_point, sub_num])  # save t value of single beta
    p_array = np.zeros(beta_t_array.shape)

    for tp in np.arange(time_point):
        for sub_index in np.arange(sub_num):

            if normalization:
                normal_bhv_rdm = np.zeros(bhv_rdm[:, sub_index].T.shape)
                for b in np.arange(m):
                    normal_bhv_rdm[:, b] = bhv_rdm[b, sub_index].T / np.linalg.norm(bhv_rdm[b, sub_index].T)

                X = np.concatenate((normal_bhv_rdm, np.ones([n, 1])), axis=1)
                Y = eeg_rdm[tp, sub_index, :] / np.linalg.norm(eeg_rdm[tp, sub_index, :])
            else:
                X = np.concatenate((bhv_rdm[:, sub_index].T, np.ones([n, 1])), axis=1)
                Y = eeg_rdm[tp, sub_index, :]

            beta = (np.linalg.inv(X.T @ X)) @ (X.T @ Y)
            rss = Y.T @ Y - beta.T @ X.T @ Y
            C = np.diag(np.linalg.inv(X.T @ X))
            sigma = np.sqrt(rss / (n - m))

            cov_A = (sigma ** 2) * np.linalg.inv(X.T @ X)  # calculating covariance matrix
            for cp_index, cp in enumerate(index_list):
                first, second = cp
                cov = cov_A[first, second]
                vp = np.sqrt((np.sqrt(C[first]) * sigma) ** 2 + (np.sqrt(C[second]) * sigma) ** 2 - 2 * cov)

                t_array[cp_index, tp, sub_index] = (beta[first] - beta[second]) / vp

            for mm in np.arange(m):
                beta_t_array[mm, tp, sub_index] = beta[mm]/(cov_A[mm, mm] ** 0.5)
                p_array[mm, tp, sub_index] = 2 * (1 - t.cdf(np.abs(beta[mm]/(cov_A[mm, mm] ** 0.5)), df))

            linear_model = OLS(Y, X).fit()
            r_square = linear_model.rsquared
            r_square_adj = linear_model.rsquared_adj
            bic = linear_model.bic
            aic = linear_model.aic

            summary = linear_model.summary()
            p_jb = float(summary.tables[0].data[3][3])

            if p_jb < 0.05 and warn_opt:
                warnings.warn("the residual variance of linear regression model is not Gaussian-distributed",
                              category=RuntimeWarning)

    return beta_t_array, p_array, t_array, index_list, r_square, r_square_adj, bic, aic


def get_tri_matrix(rdm):

    """

    Args:
        rdm: the representational dissimilarity matrix

    Returns:
        the upper triangle part of rdm
    """
    rdm_shape = rdm.shape[0]
    tri_matrix = np.zeros([rdm_shape, rdm_shape], dtype='bool')

    for row in np.arange(rdm_shape):
        for col in np.arange(rdm_shape):
            if row > col:
                tri_matrix[row, col] = True

    matrix_array = rdm[tri_matrix]

    return matrix_array


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


def normalization(array):
    return array / np.linalg.norm(array)


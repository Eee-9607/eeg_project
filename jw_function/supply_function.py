# -*- coding = utf-8 -*-
# @Time : 2023/11/8 下午3:04
# @Author : yangjw
# @Site : 
# @File : supply_function.py
# @Software : PyCharm

import os
import sys
import pickle

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
    :param string: string you want to show above progressbar
    :param cur: your current percentage progress.
                e.g you need to input (400/900) * 100, 400 is your current progress, 900 is the total progress.
    :param total: default parameter, making it 100.
    :return:
    """
    percent = '{:.2%}'.format(cur / total)
    sys.stdout.write('\r')
    sys.stdout.write(string + ": [%-100s] %s" % ('=' * int(cur), percent))
    sys.stdout.flush()

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


def reshape_array(a):
    shape = list(a.shape)
    re_shape = [1]
    re_shape.extend(shape)

    return a.reshape(tuple(re_shape))

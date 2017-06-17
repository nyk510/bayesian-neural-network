# coding: utf-8
__author__ = "nyk510"
"""
人工データ・セットを作成するスクリプト
"""

import numpy as np


def func1(x):
    """
    人工データの正しい関数例その1
    
    :param np.array x: 
    :return: 
    :rtype: np.array
    """
    return x + np.sin(5 * x) - .8


def func2(x):
    """
    人口データの正しい関数その2
    :param np.ndarray x:
    :return:
    :rtype: np.ndarray
    """
    return np.sin(5 * x) * np.abs(x)


def make_data(size, function_id=1, seed=1):
    """
    人工データの作成
    
    :param int size: 
    :param int function_id: 
    :param int seed: 
    :return: データと正しい関数の集合
    :rtype: tuple[np.array, np.array, function]
    """
    np.random.seed(seed)
    x = np.sort(np.random.uniform(-1.5, 1.5, size=size)).astype(np.float32).reshape(-1, 1)
    f = None
    if function_id == 1:
        f = func1
    elif function_id == 2:
        f = func2
    else:
        # 別の関数で試したい場合は適当にここで指定する
        raise ValueError

    y = f(x) + np.random.normal(loc=0, scale=.1, size=x.shape)
    y = y.astype(np.float32)
    return x, y, f

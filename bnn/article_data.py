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


def make_data(size, function_id=1, seed=1):
    """
    人工データの作成
    
    :param int size: 
    :param int function_id: 
    :param int seed: 
    :return: 
    :rtype: tuple of (np.array, np.array)
    """
    np.random.seed(seed)
    x = np.sort(np.random.uniform(-1.1, 1.1, size=100)).astype(np.float32)
    f = None
    if function_id == 1:
        f = func1
    else:
        # 別の関数で試したい場合は適当にここで指定する
        raise ValueError

    y = f(x) + np.random.normal(loc=.1, scale=.1, size=x.shape).astype(np.float32)
    return x, y, f

# coding: utf-8
__author__ = "nyk510"
"""
人工データ・セットを作成するスクリプト
"""

import numpy as np


class ArtificialData(object):
    """
    人口データを作成するクラス
    """

    def __init__(self, n_samples):
        self.n_samples = n_samples

    def true_function(self, x):
        """
        ノイズのない正しいデータを返す関数
        :param x:
        :return:
        :rtype: np.ndarray
        """
        raise NotImplementedError

    def generate(self, x):
        raise NotImplementedError


class Art1(ArtificialData):
    def __init__(self, n_samples, noise_scale=.1):
        self.noise_scale = noise_scale
        super().__init__(n_samples)

    def true_function(self, x):
        return func1(x)

    def generate(self, x):
        y = self.true_function(x)
        y += np.random.normal(loc=0, scale=self.noise_scale, size=x.shape)
        return y


def func1(x):
    """
    人工データの正しい関数例その1
    
    :param np.array x: 
    :return: 
    :rtype: np.array
    """
    return x + np.sin(5 * x)


def func2(x):
    """
    人口データの正しい関数その2
    :param np.ndarray x:
    :return:
    :rtype: np.ndarray
    """
    return np.sin(5 * x) * np.abs(x)


def make_data(size, function_type="art1", seed=1):
    """
    人工データの作成
    
    :param int size: 
    :param str function_type:
    :param int seed: 
    :return: データと正しい関数の集合
    :rtype: tuple[np.array, np.array, function]
    """
    np.random.seed(seed)
    x = np.sort(np.random.uniform(-1.5, 1.5, size=size)).astype(np.float32).reshape(-1, 1)
    f = None
    function_id = int(function_type[-1])
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

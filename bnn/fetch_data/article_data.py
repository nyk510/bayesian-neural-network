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

    def __init__(self, n_samples=100, noise_scale=.1):
        self.n_samples = n_samples
        self.noise_scale = noise_scale

    def true_function(self, x):
        """
        ノイズのない正しいデータを返す関数
        :param x:
        :return:
        :rtype: np.ndarray
        """
        raise NotImplementedError

    def make_x(self):
        return np.sort(np.random.uniform(-1.5, 1.5, size=self.n_samples)).astype(np.float32).reshape(-1, 1)

    def make_noise(self, x):
        return np.random.normal(loc=0, scale=self.noise_scale, size=x.shape)

    def generate(self):
        x = self.make_x()
        y = self.true_function(x)
        y += self.make_noise(y)
        return x, y


class Art1(ArtificialData):
    def true_function(self, x):
        return func1(x)


class Art2(ArtificialData):
    def true_function(self, x):
        return func2(x)

    def make_x(self):
        x1 = np.random.uniform(-1.5, -.5, size=int(self.n_samples / 2))
        x2 = np.random.uniform(.5, 1.5, size=self.n_samples - x1.shape[0])
        x = np.vstack((x1, x2)).reshape(-1, 1)
        return np.sort(x)


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

    f = None
    function_id = int(function_type[-1])
    sample_maker = None
    if function_id == 1:
        sample_maker = Art1(size)
    elif function_id == 2:
        sample_maker = Art2(size)
    else:
        # 別の関数で試したい場合は適当にここで指定する
        raise ValueError
    x, y = sample_maker.generate()
    return x, y, sample_maker.true_function

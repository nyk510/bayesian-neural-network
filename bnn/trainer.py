# coding: utf-8
__author__ = "nyk510"
"""
BNNの訓練クラスの定義
"""

from chainer.optimizer import WeightDecay
from chainer import optimizers
from chainer import Variable
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

from .bnn import BNN

import chainer.functions as F


class Transformer(object):
    """
    変数変換の実行クラス
    初めて変数が与えられたとき, 変換と同時にスケーリングのパラメータを学習し保存します。
    二回目以降は、一度目で学習したパラメータを用いて変換を行います。
    """

    def __init__(self, transform_log=False, scaling=False):
        """
        コンストラクタ
        :param bool transform_log: 目的変数をログ変換するかのbool. 
        :param bool scaling: 
        """
        self.done_fitting = False
        self.transform_log = transform_log
        self.scaling = scaling
        if scaling:
            self.scaler = StandardScaler()

    def _scaling(self, x):
        """

        :param numpy.ndarray x: 変換する変数配列 
        :return: 
        :rtype: numpy.ndarray
        """
        shape = x.shape
        if len(shape) == 1:
            x = x.reshape(-1, 1)

        if self.done_fitting:
            x = self.scaler.transform(x)
        else:
            x = self.scaler.fit_transform(x)
        x = x.reshape(shape)
        return x

    def transform(self, x):
        """
        目的変数のリストを受け取って変換器を作成し, 変換後の値を返す
        * log変換 -> scaling変換 [-1,+1]

        :param numpy.array x: 
        :rtype: numpy.array
        """
        y_trans = x[:]
        if self.transform_log:
            y_trans = np.log(x)

        if self.scaling:
            y_trans = self._scaling(x)

        return y_trans

    def inverse_transform(self, x):
        """
        変換された値を元の値に逆変換
        :param x: np.array
        :return: np.array
        """
        y_inv = x[:]
        if self.scaling:
            y_inv = self.scaler.inverse_transform(x)

        if self.transform_log:
            y_inv = np.exp(y_inv)

        return y_inv


class PreprocessMixin(object):
    """
    特徴量の前処理を行うMixinクラス
    """

    def preprocess(self, X, y=None):
        """
        入力変数の変換
        :param numpy.ndarray X: 
        :param numpy.ndarray y: 
        :return: 変換後の変数のタプル
        :rtype: tuple of (numpy.ndarray, numpy.ndarray)
        """
        x_transformed = self.x_transformer.transform(X)
        if y is None:
            return x_transformed

        y_transformed = self.y_transformer.transform(y)
        return x_transformed, y_transformed

    def inverse_y_transform(self, y):
        """
        予測値の逆変換
        :param y: 
        :return: 
        """
        return self.y_transformer.inverse_transform(y)


class BNNEstimator(BaseEstimator, PreprocessMixin):
    """
    Bayesian Neural Network を訓練し可視化を行うクラス
    """

    def __init__(self, input_dim, output_dim, hidden_dim=512, activate="relu", mask_type="gaussian", prob=.5,
                 lengthscale=10., optimizer="adam", weight_decay=4 * 10 ** -5, apply_input=False,
                 n_samples=100, x_scaling=True, y_scaling=True):
        """

        :param BNN model: Bayesian Neural Network モデル
        :param str optimizer: optimizer を指す string.
        """
        self.model = BNN(input_dim, output_dim, hidden_dim, activate, mask_type, prob,
                         lengthscale)
        self.weight_decay = weight_decay
        self.apply_input = apply_input
        self.n_samples = n_samples
        self.x_transformer = Transformer(scaling=x_scaling)
        self.y_transformer = Transformer(scaling=y_scaling)

        if optimizer == "adam":
            self.optimizer = optimizers.Adam()

        # 画像の出力先作成
        if os.path.exists("data/figures") is False:
            os.makedirs("data/figures")

    def _verify_array_shape(self, x):
        """
        numpy.array の shapeをチェックして chainer に投げられるようにする. 

        :param np.array x: 
        :return: 
        :rtype: np.array
        """
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        if np.issubdtype(x.dtype, np.integer):
            x = x.astype(np.int32)
        elif np.issubdtype(x.dtype, np.float):
            x = x.astype(np.float32)
        else:
            x = x.astype(np.float32)
        return x

    def fit(self, X, y, x_test=None, n_epoch=1000, batch_size=20, freq_print_loss=10, freq_plot=50):
        """
        モデルの最適化の開始

        :param np.array X: 
        :param np.array y: 
        :param n_epoch: 
        :param batch_size: 
        :param freq_print_loss: 
        :param freq_plot: 
        :return: 
        """

        X, y = self.preprocess(X, y)
        x_test = self.x_transformer.transform(x_test)

        N = X.shape[0]
        X = Variable(self._verify_array_shape(X))
        y = Variable(self._verify_array_shape(y))
        self.x_test = self._verify_array_shape(x_test)

        self.optimizer.setup(self.model)
        self.optimizer.add_hook(WeightDecay(self.weight_decay))
        list_loss = []

        for e in range(1, n_epoch + 1):
            perm = np.random.permutation(N)
            for i in range(0, N, batch_size):
                idx = perm[i: i + batch_size]
                _x = X[idx]
                _y = y[idx]
                self.model.zerograds()
                loss = F.mean_squared_error(self.model(_x, apply_input=self.apply_input), _y)
                loss.backward()
                self.optimizer.update()

            l = F.mean_squared_error(self.model(X, False, False), y).data
            if e % freq_print_loss == 0:
                print("epoch: {e}\tloss:{l}".format(**locals()))

            if e % freq_plot == 0:
                fig, ax = self.plot_posterior(x_test, X.data, y.data, n_samples=self.n_samples)
                ax.set_title("epoch: {0}".format(e))
                fig.tight_layout()
                s_condition = self.model.pretty_string()
                fig.savefig("data/figures/epoch={e:04d}_{s_condition}.png".format(**locals()), dpi=150)
                plt.close("all")
            list_loss.append([e, l])

        plot_logloss(list_loss, self.model.pretty_string())

    def plot_posterior(self, x_test, x_train=None, y_train=None, n_samples=100):
        model = self.model
        xx = self.x_transformer.inverse_transform(x_test)
        x_train, y_train = self.x_transformer.inverse_transform(x_train), self.inverse_y_transform(y_train)
        predict_values = self.posterior(xx, n=n_samples)
        predict_values = np.array(predict_values)

        predict_mean = predict_values.mean(axis=0)
        predict_var = predict_values.var(axis=0)
        tau = 1 ** 2 * (1 - model.mask.prob) / (2 * len(x_train) * 4 * 10 ** -3)
        predict_var += tau ** -1

        fig = plt.figure(figsize=(8, 5))
        ax1 = fig.add_subplot(111)
        ax1.plot(x_train[:, 0], y_train[:, 0], "o", alpha=.3, color="C0", label="Training Data Points")
        for i in range(100):
            if i == 0:
                ax1.plot(xx[:, 0], predict_values[i], color="C1", alpha=.05, label="Posterior Samples")
            else:
                ax1.plot(xx[:, 0], predict_values[i], color="C1", alpha=.05)
        ax1.plot(xx[:, 0], predict_mean, "--", color="C1", label="Posterior Mean")
        # ax1.set_ylim(-3., 1.5)
        # ax1.set_xlim(-2, 2)
        ax1.legend(loc=4)
        return fig, ax1

    def posterior(self, x, n=3):
        """

        :param np.array x: 
        :param int n: 
        :return: 
        """
        x = self._verify_array_shape(x)
        x = self.preprocess(x)
        x = Variable(x)
        pred = [self.model(x, apply_input=False, apply_hidden=True).data.reshape(-1) for _ in range(n)]
        pred = [self.inverse_y_transform(p) for p in pred]
        return pred


def plot_logloss(loss, name, save=True):
    loss = np.array(loss)
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)
    ax1.plot(loss[:, 1], color="C0")
    ax1.set_yscale("log")
    if save:
        fig.savefig("{name}.png".format(**locals()), dpi=150)
    return

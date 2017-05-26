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

from .bnn import BNN

import chainer.functions as F


class Trainer(object):
    """
    Bayesian Neural Network を訓練し可視化を行うクラス
    """

    def __init__(self, model, optimizer="adam", weight_decay=4 * 10 ** -5, apply_input=False,
                 n_samples=100):
        """

        :param BNN model: Bayesian Neural Network モデル
        :param str optimizer: optimizer を指す string.
        """
        self.model = model
        self.weight_decay = weight_decay
        self.apply_input = apply_input
        self.n_samples = n_samples

        if optimizer == "adam":
            self.optimizer = optimizers.Adam()

        # 画像の出力先作成
        if os.path.exists("data/figures") is False:
            os.makedirs("data/figures")

    def _verify_array_shape(self, x):
        """
        numpy.array の shapeをチェックして chainer に投げられるようにする. 

        :param np.ndarray x: 
        :return: 
        :rtype: Variable
        """
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        if np.issubdtype(x.dtype, np.integer):
            x = x.astype(np.int32)
        elif np.issubdtype(x.dtype, np.float):
            x = x.astype(np.float32)
        else:
            raise ValueError
        return x

    def fit(self, X, y, n_epoch=1000, batch_size=20, freq_print_loss=10, freq_plot=50):
        """
        モデルの最適化の開始

        :param np.ndarray X: 
        :param np.ndarray y: 
        :param n_epoch: 
        :param batch_size: 
        :param freq_print_loss: 
        :param freq_plot: 
        :return: 
        """

        N = X.shape[0]

        X = Variable(self._verify_array_shape(X))
        y = Variable(self._verify_array_shape(y))

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
                fig, ax = self.plot_posterior(X.data, y.data, n_samples=self.n_samples)
                ax.set_title("epoch: {0}".format(e))
                fig.tight_layout()
                s_condition = self.model.pretty_string()
                fig.savefig("data/figures/epoch={e:04d}_{s_condition}.png".format(**locals()), dpi=150)
                plt.close("all")
            list_loss.append([e, l])

        plot_logloss(list_loss, self.model.pretty_string())

    def plot_posterior(self, x_train, y_train, n_samples=100):
        model = self.model
        xx = np.linspace(-2., 2, 200, dtype=np.float32)
        predict_values = [self.model(Variable(xx).reshape(-1, 1), apply_input=False, apply_hidden=True).data.reshape(-1)
                          for
                          i in range(n_samples)]
        predict_values = np.array(predict_values)

        predict_mean = predict_values.mean(axis=0)
        predict_var = predict_values.var(axis=0)
        tau = 1 ** 2 * (1 - model.mask.prob) / (2 * len(x_train) * 4 * 10 ** -3)
        predict_var += tau ** -1

        fig = plt.figure(figsize=(8, 5))
        ax1 = fig.add_subplot(111)
        ax1.plot(x_train, y_train, "o", alpha=.3, color="C0", label="Training Data Points")
        for i in range(100):
            if i == 0:
                ax1.plot(xx, predict_values[i], color="C1", alpha=.05, label="Posterior Samples")
            else:
                ax1.plot(xx, predict_values[i], color="C1", alpha=.05)
        ax1.plot(xx, predict_mean, "--", color="C1", label="Posterior Mean")
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
        x = Variable(x.astype(np.float32).reshape(-1, 1))
        pred = [self.model(x, apply_input=False, apply_hidden=True).data.reshape(-1) for _ in range(n)]
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

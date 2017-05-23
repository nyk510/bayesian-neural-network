# coding: utf-8
__author__ = "nyk510"
"""
BNNの訓練クラスの定義
"""

from chainer.optimizer import WeightDecay
from chainer import optimizers
from chainer import Variable
import chainer.functions as F
import numpy as np
import matplotlib.pyplot as plt
import os

from .article_data import make_data


class Trainer(object):
    """
    Bayesian Neural Network を訓練し可視化を行うクラス
    """

    def __init__(self, model, optimizer="adam", train_size=100, train_type_id=1, seed=1):
        """
        
        :param bnn.BNN model: Bayesian Neural Network モデル
        :param str optimizer: optimizer を指す string.
        :param int train_size: training data のサイズ
        :param int train_type_id: training data 作成の際の真の関数のid
        :param int seed: random seed value
        """
        self.model = model

        if optimizer == "adam":
            self.optimizer = optimizers.Adam()

        self.x_train, self.y_train, self.true_function = make_data(size=train_size, function_id=train_type_id, seed=seed)
        # 画像の出力先作成
        if os.path.exists("figures") is False:
            os.makedirs("figures")

    def run(self, n_epoch=1000, batch_size=20, weight_decay=4 * 10 ** -5, freq_print_loss=10, freq_plot=50,
            n_samples=100):
        N = len(self.x_train)
        X = Variable(self.x_train.reshape(-1, 1))
        y = Variable(self.y_train.reshape(-1, 1))

        self.optimizer.setup(self.model)
        self.optimizer.add_hook(WeightDecay(weight_decay))
        list_loss = []

        for e in range(n_epoch):
            perm = np.random.permutation(N)
            for i in range(0, N, batch_size):
                idx = perm[i: i + batch_size]
                _x = X[idx]
                _y = y[idx]
                self.model.zerograds()
                loss = F.mean_squared_error(self.model(_x), _y)
                loss.backward()
                self.optimizer.update()

            l = F.mean_squared_error(self.model(X, False), y).data
            if e % 10 == 0:
                print("epoch: {e}\tloss:{l}".format(**locals()))

            if e % 50 == 0:
                fig, _ = self.plot_posterior(n_samples=n_samples)
                s_condition = self.model.pretty_string()
                fig.savefig("figures/epoch={e}_{s_condition}.png".format(**locals()), dpi=150)
                plt.close("all")
            list_loss.append([e, l])

        plot_logloss(list_loss, self.model.pretty_string())

    def plot_posterior(self, n_samples=100):
        model = self.model
        x_train, y_train = self.x_train, self.y_train
        xx = np.linspace(-2., 2, 200, dtype=np.float32)
        predict_values = [model(Variable(xx).reshape(-1, 1), True, False).data.reshape(-1) for i in range(n_samples)]
        predict_values = np.array(predict_values)

        predict_mean = predict_values.mean(axis=0)
        predict_var = predict_values.var(axis=0)
        tau = 1 ** 2 * (1 - model.mask.prob) / (2 * len(x_train) * 4 * 10 ** -3)
        predict_var += tau ** -1

        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(111)
        ax1.plot(x_train, y_train, "o", alpha=.3, color="C0", label="Training data point")
        ax1.plot(xx, self.true_function(xx), color="C0", label="True Function")
        for i in range(100):
            if i == 0:
                ax1.plot(xx, predict_values[i], color="C1", alpha=.05, label="Posterior Samples")
            else:
                ax1.plot(xx, predict_values[i], color="C1", alpha=.05)
        ax1.plot(xx, predict_mean, color="C1", label="Posterior mean")
        ax1.set_ylim(-3., 1.5)
        ax1.legend(loc=4)
        fig.tight_layout()
        return fig, ax1


def plot_logloss(loss, name, save=True):
    loss = np.array(loss)
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)
    ax1.plot(loss[:, 1], color="C0")
    ax1.set_yscale("log")
    if save:
        fig.savefig("{name}.png".format(**locals()), dpi=150)
    return

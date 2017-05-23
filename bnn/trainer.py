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

from .bnn import BNN
from .article_data import make_data


class Trainer(object):
    """
    Bayesian Neural Network を訓練し可視化を行うクラス
    """

    def __init__(self, model, optimizer="adam", train_size=100, train_type_id=1, seed=1):
        """
        
        :param BNN model: Bayesian Neural Network モデル
        :param str optimizer: optimizer を指す string.
        :param int train_size: training data のサイズ
        :param int train_type_id: training data 作成の際の真の関数のid
        :param int seed: random seed value
        """
        self.model = model

        if optimizer == "adam":
            self.optimizer = optimizers.Adam()

        self.x_train, self.y_train, self.true_function = make_data(size=train_size, function_id=train_type_id,
                                                                   seed=seed)
        # 画像の出力先作成
        if os.path.exists("figures") is False:
            os.makedirs("figures")

    def run(self, n_epoch=1000, batch_size=20, weight_decay=4 * 10 ** -5, apply_input=False,
            n_samples=100, freq_print_loss=10, freq_plot=50):
        """
        
        :param n_epoch: 
        :param batch_size: 
        :param weight_decay: 
        :param freq_print_loss: 
        :param freq_plot: 
        :param apply_input: 
        :param n_samples: 
        :return: 
        """
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.apply_input = apply_input
        self.n_samples = n_samples

        N = len(self.x_train)
        X = Variable(self.x_train.reshape(-1, 1))
        y = Variable(self.y_train.reshape(-1, 1))

        self.optimizer.setup(self.model)
        self.optimizer.add_hook(WeightDecay(weight_decay))
        list_loss = []

        for e in range(1, n_epoch + 1):
            perm = np.random.permutation(N)
            for i in range(0, N, batch_size):
                idx = perm[i: i + batch_size]
                _x = X[idx]
                _y = y[idx]
                self.model.zerograds()
                loss = F.mean_squared_error(self.model(_x, apply_input=apply_input), _y)
                loss.backward()
                self.optimizer.update()

            l = F.mean_squared_error(self.model(X, False, False), y).data
            if e % freq_print_loss == 0:
                print("epoch: {e}\tloss:{l}".format(**locals()))

            if e % freq_plot == 0:
                fig, ax = self.plot_posterior(n_samples=n_samples)
                ax.set_title("epoch: {0}".format(e))
                fig.tight_layout()
                s_condition = self.model.pretty_string()
                fig.savefig("figures/epoch={e:04d}_{s_condition}.png".format(**locals()), dpi=150)
                plt.close("all")
            list_loss.append([e, l])

        plot_logloss(list_loss, self.model.pretty_string())

    def plot_posterior(self, n_samples=100):
        model = self.model
        x_train, y_train = self.x_train, self.y_train
        xx = np.linspace(-2., 2, 200, dtype=np.float32)
        predict_values = [model(Variable(xx).reshape(-1, 1), apply_input=False, apply_hidden=True).data.reshape(-1) for
                          i in range(n_samples)]
        predict_values = np.array(predict_values)

        predict_mean = predict_values.mean(axis=0)
        predict_var = predict_values.var(axis=0)
        tau = 1 ** 2 * (1 - model.mask.prob) / (2 * len(x_train) * 4 * 10 ** -3)
        predict_var += tau ** -1

        fig = plt.figure(figsize=(8, 5))
        ax1 = fig.add_subplot(111)
        ax1.plot(x_train, y_train, "o", alpha=.3, color="C0", label="Training Data Points")
        ax1.plot(xx, self.true_function(xx), color="C0", label="True Function")
        for i in range(100):
            if i == 0:
                ax1.plot(xx, predict_values[i], color="C1", alpha=.05, label="Posterior Samples")
            else:
                ax1.plot(xx, predict_values[i], color="C1", alpha=.05)
        ax1.plot(xx, predict_mean, "--", color="C1", label="Posterior Mean")
        ax1.set_ylim(-3., 1.5)
        ax1.set_xlim(-2, 2)
        ax1.legend(loc=4)
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

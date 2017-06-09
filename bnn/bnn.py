# coding: utf-8
__author__ = "nyk510"
"""
ベイジアンニューラルネットワークの定義
"""

from chainer import Chain
from chainer import Variable
import chainer.functions as F
import chainer.links as L
import numpy as np


class Mask(object):
    """
    入力変数に数値を掛けて摂動を与える関数クラス
    """

    def __init__(self, name="dropout", prob=.5):
        """
        :param str name: マスクに用いる手法の名前. "dropout", "gaussian", None のいずれかを指定
        :param float prob: マスクの確率
        """

        self.prob = self._check_prob(prob)
        self.name = name
        if name == "dropout":
            self.mask_generator = self._dropout_mask
        elif name == "gaussian":
            self.mask_generator = self._gaussian_mask
        elif name is None or name.lower() == "none":
            self.mask_generator = self._none_mask
        else:
            raise NameError("name: {name} に該当するmask関数が見当たりません. ".format(**locals()))

    def __repr__(self):
        s = "maskname={0.name}_prob={0.prob}".format(self)
        return s

    def _check_prob(self, prob):
        if prob >= 1.:
            prob = 1.
        elif prob < 0:
            prob = 0
        return prob

    def _dropout_mask(self, size):
        z = np.random.binomial(1, self.prob, size=size).astype(np.float32) * self.prob ** -1
        return z

    def _gaussian_mask(self, size):
        sigma = self.prob / (1. - self.prob)
        z = np.random.normal(loc=1., scale=sigma, size=size).astype(np.float32)
        return z

    def _none_mask(self, size):
        return np.array([1.] * size).astype(np.float32)

    def _make(self, size):
        z = self.mask_generator(size)
        return Variable(np.diag(z))

    def apply(self, h, do_mask=True):
        """
        ベクトルにマスクを掛ける関数
        :param Variable h: マスクされる変数
        :return: masked variable
        :rtype: Variable
        """
        if do_mask is False:
            return h

        size = h.shape[1]
        z = self._make(size)
        z = F.matmul(h, z)
        return z


class BNN(Chain):
    """
    ベイジアンニューラルネットの重み学習を行うクラス
    """

    def __init__(self, input_dim, output_dim,
                 hidden_dim=512, activate="relu", mask_type="gaussian", prob=.5, lengthscale=10.):
        """
        :param int input_dim: 入力層の次元数
        :param int output_dim: 出力層の次元数
        :param int hidden_dim: 隠れ層の次元数
        :param str activate: 活性化関数
        :param str mask_type: 
            変数へのマスクの種類を表すstring. 
            "dropout", "gaussian", Noneのいずれかを指定
        :param float prob: 
            dropoutの確率を表すfloat. 
            0.のときdropoutをしないときに一致します. 
            [0, 1) の小数
        :param float lengthscale:
            初期のネットワーク重みの精度パラメータ. 大きい値になるほど0に近い値を取ります. 
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activate_name = activate
        self.activate = self._get_function(activate)
        self.mask_type = mask_type
        self.lengthscale = lengthscale

        super().__init__(
            l1=L.Linear(input_dim, hidden_dim,
                        initial_bias=np.random.normal(scale=1. / lengthscale, size=(hidden_dim)),
                        initialW=np.random.normal(scale=1. / lengthscale, size=(hidden_dim, input_dim))),
            l2=L.Linear(hidden_dim, hidden_dim,
                        initial_bias=np.random.normal(scale=1. / lengthscale, size=(hidden_dim)),
                        initialW=np.random.normal(scale=1. / lengthscale, size=(hidden_dim, hidden_dim))),
            l3=L.Linear(hidden_dim, hidden_dim,
                        initial_bias=np.random.normal(scale=1. / lengthscale, size=(hidden_dim)),
                        initialW=np.random.normal(scale=1. / lengthscale, size=(hidden_dim, hidden_dim))),
            l4=L.Linear(hidden_dim, output_dim,
                        initial_bias=np.random.normal(scale=1. / lengthscale, size=(output_dim)),
                        initialW=np.random.normal(scale=1. / lengthscale, size=(output_dim, output_dim)))
        )

        self.mask = Mask(name=mask_type, prob=prob)

    def _get_function(self, s):
        """
        文字列からそれに対応する関数を取得
        
        :param str s: 関数を表す文字列
        :return: 
        """
        if s == "relu":
            f = F.relu
        elif s == "sigmoid":
            f = F.sigmoid
        elif s == "tanh":
            f = F.tanh
        else:
            print("対応する関数が見つかりません")
            f = lambda x: x
        return f

    def __call__(self, x, apply_input=False, apply_hidden=True):
        """
        ネットワークの出力を作成
        
        :param Variable x: 入力ベクトル
        :param bool apply_hidden: 
            隠れ層に対してマスクをかけるかのフラグ. 
            True のときm `mask` によって生成されたマスクを隠れ層に掛ける
        :param bool apply_input:
            入力層に対してマスクをかけるかのフラグ. 
            True にすると学習が不安定になることが観測されているため, 学習時には False が推奨
        :return: 出力
        :rtype: Variable
        """
        x1 = self.mask.apply(x, apply_input)
        h1 = self.activate(self.l1(x1))

        h1 = self.mask.apply(h1, apply_hidden)
        h2 = self.activate(self.l2(h1))

        h2 = self.mask.apply(h2, apply_hidden)
        h3 = self.activate(self.l3(h2))

        h3 = self.mask.apply(h3, apply_hidden)
        h4 = self.l4(h3)
        return h4

    def pretty_string(self):
        """
        ネットワークの条件をいい感じの文字列にして返す
        
        :return: ネットワーク条件の文字
        :rtype: str
        """
        s = "hidden={0.hidden_dim}_activate={0.activate_name}_{0.mask}".format(self)
        return s

# coding: utf-8
__author__ = "nyk510"
"""
事前分布からサンプリングされたニューラルネットワークの出力可視化スクリプト
"""

import numpy as np
from chainer import Variable
import matplotlib.pyplot as plt
import os

from bnn.setting import DATA_DIR
from bnn.bnn import BNN

output_dir = os.path.join(DATA_DIR, 'priors')
os.makedirs(output_dir, exist_ok=True)

def main():    
    list_activate = ["relu", "tanh", "sigmoid"]

    data = []
    for f_name in list_activate:
        list_y = []
        for i in range(20):
            bnn = BNN(1, 1, activate=f_name, hidden_dim=1000, lengthscale=.1)
            x = Variable(np.linspace(-2., 2., 100).reshape(-1, 1).astype(np.float32))
            y = bnn(x, False).data
            list_y.append(y.reshape(-1))
        list_y = np.array(list_y)
        data.append(
            {
                "title": bnn.__str__(),
                "name": f_name,
                "y": list_y
            }
        )
    for i, d in enumerate(data):
        fig = plt.figure(figsize=(5, 5))
        ax_i = fig.add_subplot(111)
        list_y = d["y"]
        for y_i in list_y:
            ax_i.plot(x.data.reshape(-1), y_i, color="C0")
        ax_i.set_title("活性化関数: {name}".format(**d))
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "prior_{title}.png".format(**d)), dpi=150)


if __name__ == "__main__":
    main()

# coding: utf-8
__author__ = "nyk510"
"""
ドロップアウトを用いたニューラルネットワークの学習が, 重みに対するベイズ学習になっており
学習済みのニューラルネットワークから, 重みの事後分布をサンプルできることを確認する
"""

from bnn import BNNEstimator
from bnn import fetch_data
import argparse


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-e", type=int, default=1000, help="number of epochs.")
    p.add_argument("-m", "--mask", default="dropout", type=str, help="type of mask function. dropout, gaussian, none.")
    p.add_argument("-a", "--activate", default="relu", type=str, help="type of activate function.")
    p.add_argument("-d", "--data", default="nikkei", type=str)
    p.add_argument("--hidden", type=int, default=512, help="number of hidden dimensions. ")
    args = p.parse_args()
    return args


def main(model_params, train_params):
    """
    
    :param dict model_params: 
    :param dict train_params: 
    :return: 
    """
    clf = BNNEstimator(**model_params)
    clf.fit(**train_params)
    return


if __name__ == "__main__":
    # main()
    args = parser()
    if args.data == "nikkei":
        x_train, y_train, x_test = fetch_data.fetch_nikkei()

    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]

    model_params = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "mask_type": args.mask,
        "activate": args.activate,
        "hidden_dim": args.hidden
    }

    train_params = {
        "n_epoch": args.e,
        "X": x_train,
        "y": y_train,
        "x_test": x_test,
    }
    print(args)

    main(model_params, train_params)

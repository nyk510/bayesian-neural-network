# coding: utf-8
__author__ = "nyk510"
"""
ドロップアウトを用いたニューラルネットワークの学習が, 重みに対するベイズ学習になっており
学習済みのニューラルネットワークから, 重みの事後分布をサンプルできることを確認する
"""

from bnn import BNN, Trainer
import argparse


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-e", type=int, default=1000, help="number of epochs. ")
    p.add_argument("-m", "--mask", default="dropout", type=str, help="type of mask function. dropout, gaussian, none. ")
    p.add_argument("-a", "--activate", default="relu", type=str, help="type of activate function. ")
    p.add_argument("--hidden", type=int, default=512, help="number of hidden dimensions. ")
    args = p.parse_args()
    return args


def main(model_params, train_params):
    """
    
    :param dict model_params: 
    :param dict train_params: 
    :return: 
    """
    model = BNN(1, 1, **model_params)
    trainer = Trainer(model)
    trainer.run(**train_params)
    return


if __name__ == "__main__":
    # main()
    args = parser()
    model_params = {
        "mask_type": args.mask,
        "activate": args.activate,
        "hidden_dim": args.hidden
    }
    train_params = {
        "n_epoch": args.e
    }
    print(args)

    main(model_params, train_params)

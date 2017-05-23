# coding: utf-8
__author__ = "nyk510"
"""
ドロップアウトを用いたニューラルネットワークの学習が, 重みに対するベイズ学習になっており
学習済みのニューラルネットワークから, 重みの事後分布をサンプルできることを確認する
"""
from bnn import BNN, Trainer

def main():
    model = BNN(1, 1)
    trainer = Trainer(model)
    trainer.run()


if __name__ == "__main__":
    main()
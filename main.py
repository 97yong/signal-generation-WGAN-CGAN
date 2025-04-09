from arguments import get_args
from dataset import get_dataloaders
from train import train_model
from test import evaluate
from preprocess import CWRUPreprocessor
import os
import numpy as np

def main():
    opt = get_args()

    data_path = './data'
    data_list = ['97.mat', '105.mat', '118.mat', '130.mat']
    data_info = ['X097_DE_time', 'X105_DE_time', 'X118_DE_time', 'X130_DE_time']
    condition_list = ['Normal', 'IR', 'B', 'OR']

    preprocessor = CWRUPreprocessor(
        data_dir=data_path,
        data_list=data_list,
        data_info=data_info,
        condition_list=condition_list
    )

    X, Y = preprocessor.get_data()

    train_dl, valid_dl, test_dl = get_dataloaders(X, Y, opt)

    model, result_dict = train_model(opt, train_dl, valid_dl)
    acc = evaluate(test_dl, model)
    print(f"✅ Test Accuracy: {acc:.2f}%")


if __name__ == '__main__':
    main()
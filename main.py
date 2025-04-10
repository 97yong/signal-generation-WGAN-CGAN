from arguments import get_args
from dataset import get_dataloaders
from preprocess import CWRUPreprocessor
from train import train_cgan
from test import evaluate_cgan
import torch
import numpy as np
import random

def main():
    opt = get_args()
    
    random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    # torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    
    data_path = './data'
    data_list = ['97.mat', '105.mat', '118.mat', '130.mat']
    data_info = ['X097_DE_time', 'X105_DE_time', 'X118_DE_time', 'X130_DE_time']
    condition_list = ['Normal', 'IR', 'B', 'OR']

    preprocessor = CWRUPreprocessor(data_dir=data_path, data_list=data_list,
                                     data_info=data_info, condition_list=condition_list, window_size=opt.sample_len)
    X, Y = preprocessor.get_data()
    train_dl, test_dl = get_dataloaders(X, Y, opt)

    G, D = train_cgan(opt, train_dl)
    scores_df, stats_summary, fake_signals = evaluate_cgan(test_dl, G, sample_len=opt.sample_len, sr=opt.sr)
    print(stats_summary)

if __name__ == '__main__':
    main()

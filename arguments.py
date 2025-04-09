import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lamda', type=float, default=0.97)
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--train_size', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=64)
    
    return parser.parse_args('')

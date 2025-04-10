import argparse

def get_args():
    parser = argparse.ArgumentParser(description="WGAN for Fault Diagnosis (CWRU)")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_size', type=float, default=0.8)
    parser.add_argument('--gan_weight', type=float, default=1.0)
    parser.add_argument('--clf_weight', type=float, default=1.0)
    parser.add_argument('--clip_value', type=float, default=0.01)
    parser.add_argument('--n_critic', type=int, default=5)
    
    parser.add_argument('--sample_len', type=int, default=1200)
    parser.add_argument('--sr', type=int, default=12000) # Sampling Rate
    return parser.parse_args()

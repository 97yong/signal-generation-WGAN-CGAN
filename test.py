import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import seaborn as sns

def evaluate_cgan(real_dl, generator, device='cuda', save_dir='./outputs', num_classes=4, sample_len=1200, sr=12000):

    os.makedirs(save_dir, exist_ok=True)
    generator.eval()
    real_X, real_Y = [], []

    with torch.no_grad():
        for x, y in real_dl:
            real_X.append(x)
            real_Y.append(y)

    real_X = torch.cat(real_X).squeeze(1).cpu().numpy()  # shape: (N, 1200)
    real_Y = torch.cat(real_Y).cpu().numpy()             # shape: (N,)

    # Create conditional noise and one-hot labels
    batch_size = real_X.shape[0]
    z = torch.randn(batch_size, 100).to(device)
    labels = torch.tensor(real_Y, dtype=torch.long).to(device)
    one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()

    # Generate fake signals
    gen_X = generator(z, one_hot).detach().cpu().numpy()

    # Compute FFT Cosine Similarity
    cosine_scores = []
    for cls in range(num_classes):
        real_cls = real_X[real_Y == cls]
        gen_cls = gen_X[real_Y == cls]

        real_cls = real_cls.reshape(real_cls.shape[0], -1)
        gen_cls = gen_cls.reshape(gen_cls.shape[0], -1)

        fft_real = np.abs(fft(real_cls, axis=1))[:, :sample_len // 2]
        fft_gen = np.abs(fft(gen_cls, axis=1))[:, :sample_len // 2]

        sim = cosine_similarity(fft_real, fft_gen).diagonal()
        for s in sim:
            cosine_scores.append({'Class': cls, 'CosineSimilarity': s})

    df_scores = pd.DataFrame(cosine_scores)
    summary = df_scores.groupby('Class').agg(['mean', 'std']).round(4)

    # 📊 Boxplot
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Class', y='CosineSimilarity', data=df_scores, palette='Set2')
    plt.title('FFT Cosine Similarity (Generated vs Real)')
    plt.savefig(f'{save_dir}/fft_similarity_boxplot.png', bbox_inches='tight')
    plt.close()

    # 🎨 Example FFT comparison
    N = sample_len
    freqs = np.fft.fftfreq(N, d=1 / sr)[:N // 2]
    fig, axs = plt.subplots(num_classes, 2, figsize=(10, 10))
    for cls in range(num_classes):
        idx = np.where(real_Y == cls)[0][0]
        real_sample = real_X[idx].squeeze()
        gen_sample = gen_X[idx].squeeze()

        fft_real = np.abs(fft(real_sample))[:N // 2]
        fft_gen = np.abs(fft(gen_sample))[:N // 2]

        axs[cls, 0].plot(freqs, fft_real)
        axs[cls, 0].set_title(f'Class {cls} - Real')
        axs[cls, 1].plot(freqs, fft_gen)
        axs[cls, 1].set_title(f'Class {cls} - Generated')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/fft_comparison_samples.png', bbox_inches='tight')
    plt.close()

    return df_scores, summary, gen_X

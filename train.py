import torch
import torch.nn as nn
import torch.optim as optim
from model import CGANGenerator, CGANDiscriminator
from tqdm import tqdm

def train_cgan(opt, train_dl, n_critic=5, clip_value=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = CGANGenerator(noise_dim=100, class_dim=4, output_dim=1200).to(device)
    D = CGANDiscriminator(input_dim=1200, class_dim=4).to(device)

    optimizer_G = optim.RMSprop(G.parameters(), lr=opt.lr)
    optimizer_D = optim.RMSprop(D.parameters(), lr=opt.lr)

    for epoch in range(opt.epochs):
        G.train(); D.train()
        loop = tqdm(enumerate(train_dl), total=len(train_dl), desc=f"Epoch {epoch+1}/{opt.epochs}")
        
        for i, (x_real, y_real) in loop:
            x_real = x_real.squeeze(1).to(device)         # [B, 1200]
            y_real_idx = y_real.to(device)                # [B]
            y_onehot = torch.nn.functional.one_hot(y_real_idx, num_classes=4).float().to(device)
            batch_size = x_real.size(0)

            # === Train Discriminator ===
            for _ in range(n_critic):
                z = torch.randn(batch_size, 100).to(device)
                x_fake = G(z, y_onehot).detach()

                d_real = D(x_real, y_onehot)
                d_fake = D(x_fake, y_onehot)

                d_loss = -torch.mean(d_real) + torch.mean(d_fake)

                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()

                # Clip weights (WGAN rule)
                for p in D.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            # === Train Generator ===
            z = torch.randn(batch_size, 100).to(device)
            x_fake = G(z, y_onehot)
            d_fake = D(x_fake, y_onehot)
            g_loss = -torch.mean(d_fake)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            loop.set_postfix(D_loss=d_loss.item(), G_loss=g_loss.item())

    return G, D

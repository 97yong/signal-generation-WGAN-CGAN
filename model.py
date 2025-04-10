import torch
import torch.nn as nn

class CGANGenerator(nn.Module):
    def __init__(self, noise_dim=100, class_dim=4, output_dim=1200):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + class_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )
    
    def forward(self, z, labels):  # labels: one-hot encoded
        x = torch.cat([z, labels], dim=1)
        return self.model(x)

class CGANDiscriminator(nn.Module):
    def __init__(self, input_dim=1200, class_dim=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + class_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
    
    def forward(self, x, labels):  # labels: one-hot encoded
        x = torch.cat([x, labels], dim=1)
        return self.model(x)

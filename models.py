import torch
import torch.nn as nn
import torch.nn.functional as F
latent_size = 64
hidden_size = 256
image_size = 784
class Generator(nn.Module):
    def __init__(self):
        super().__init()
        self.G = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, image_size),
            nn.Tanh()
        )
    def forward(self, x):
        img=self.G(x)
        return img



class Discriminator(nn.Module):
    def __init__(self):
        super().__init()
    D = nn.Sequential(
        nn.Linear(image_size,hidden_size),
        nn.LeakyReLU(0.2),
        nn.Linear(hidden_size, hidden_size),
        nn.LeakyReLU(0.2),
        nn.Linear(hidden_size,1),
        nn.Sigmoid()
    )
    def forward(self, x):
        img=self.D(x)
        return img


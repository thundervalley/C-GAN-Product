import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self.latent_size,self.hidden_size,self.image_size):
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
        img=self.Generator(x)
        return img



class Discriminator(nn.Module):
    def __init__(self.latent_size,self.hidden_size,self.image_size):
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
        img=self.Discriminator(x)
        return img


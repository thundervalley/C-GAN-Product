import torch.nn as nn


class Generator(nn.Module):
    def __init__(latent_size,hidden_size,image_size):
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
        output=self.G(x)
        return output



class Discriminator(nn.Module):
    def __init__(latent_size,hidden_size,image_size):
        super().__init()
        self.D = nn.Sequential(
            nn.Linear(image_size,hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size,1),
            nn.Sigmoid()
        )
    def forward(self, x):
        output=self.D(x)
        return output


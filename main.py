import os 
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from configparser import ConfigParser

from models import Generator, Discriminator
from trainer import train

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

latent_size = 64
hidden_size = 256
image_size = 784
num_epochs = 200
batch_size = 100
sample_dir ='samples'


def main(config):


    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5],std=[0.5])])

    mnist = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    data_loader = torch.utils.data.DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)

    D = Discriminator(
        config.getint('h_param', 'hidden_size'),
        config.getint('h_param', 'image_size' )
        )
    G = Generator(
        config.getint('h_param','latent_size'),
        config.getint('h_param', 'hidden_size'),
        config.getint('h_param', 'image_size' )
    )
    D = D.to(device)
    G = G.to(device)

    criterion = nn.BCELoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)
    
    train(data_loader,D,G,d_optimizer,g_optimizer,criterion,config,device)

    
    

if __name__ == '__main__':
    config = ConfigParser()
    config.read('parser.ini')
    main(config)
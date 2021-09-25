import os
import torch
from torchvision.utils import save_image




def train(data_loader,D,G,d_optimizer,g_optimizer,criterion,config,device = torch.device('cuda')):
    total_step = len(data_loader)
    num_epochs = config.getint('h_param','num_epochs')
    batch_size = config.getint('h_param','batch_size')
    latent_size = config.getint('h_param','latent_size')

    def denorm(x):
        out = (x+1) / 2
        return out.clamp(0,1)

    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(data_loader):
            images = images.reshape(batch_size, -1).to(device)
            
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            outputs = D(images)
            d_loss_real = criterion(outputs, real_labels)
            real_score = outputs

            z = torch.randn(batch_size, latent_size).to(device)
            fake_images = G(z)
            outputs = D(fake_images)
            d_loss_fake = criterion(outputs, fake_labels)
            fake_score = outputs

            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            z = torch.randn(batch_size, latent_size).to(device)
            fake_images = G(z)
            outputs = D(fake_images)

            g_loss = criterion(outputs, real_labels)

            d_optimizer.zero_grad()
            g_optimizer.zero_grad()

            if (i+1) % 200 == 0:
                 print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(), 
                          real_score.mean().item(), fake_score.mean().item()))
        if (epoch) == 0:
            images = images.reshape(images.size(0), 1, 28, 28)
            save_image(denorm(images), os.path.join(config.get('path','sample_dir'),'real_images.png'))

        fake_images = fake_images.reshape(fake_images.size(0),1,28,28)
        save_image(denorm(fake_images),os.path.join(config.get('path','sample_dir'),'fake_images-{}.png'.format(epoch+1)))
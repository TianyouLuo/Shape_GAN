import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, image_size, latent_size, num_chanel, hidden_size):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.main = nn.Sequential(
            nn.Linear(self.image_size, 16 * hidden_size, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Linear(16 * hidden_size, 4 * hidden_size, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Linear(4 * hidden_size, hidden_size, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, input):
        return self.main(input.view(-1, self.image_size))


class Decoder(nn.Module):
    def __init__(self, image_size, hidden_size, latent_size):
        super(Decoder, self).__init__()
        self.image_size = image_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.main = nn.Sequential(
            nn.Linear(latent_size, hidden_size, bias=False),
            nn.ReLU(True),
            nn.Linear(hidden_size,  4* hidden_size, bias=False),
            nn.ReLU(True),
            nn.Linear(4 * hidden_size, 16 * hidden_size, bias=False),
            nn.ReLU(True),
            nn.Linear(16 * hidden_size, self.image_size, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class ShapeAutoencoder(nn.Module):
    def __init__(self, image_size, hidden_size, latent_size, device):
        super(ShapeAutoencoder, self).__init__()
        self.image_size = image_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.device = device
        self.decoder = Decoder(image_size, hidden_size, latent_size)
    
    def compute_loss_WGAN(self, discriminator, optimizer, minibatch, rand_dist):
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        #y_data, _ = discriminator(data.detach())
        #y_fake, _ = discriminator(data_fake.detach())
        y_data = discriminator(data)
        y_fake = discriminator(data_fake)
        errD_real = torch.mean(y_data)
        errD_fake = torch.mean(y_fake)
        errD = -errD_real + errD_fake
        optimizer.zero_grad()
        errD.backward()
        optimizer.step()
        
        return errD
    
    def compute_loss_G(self, discriminator, minibatch, rand_dist):
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        #y_fake, _ = discriminator(data_fake.detach())
        y_fake = discriminator(data_fake)
        errG = -torch.mean(y_fake)
        return errG

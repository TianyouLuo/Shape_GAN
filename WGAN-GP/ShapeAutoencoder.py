import torch
import torch.nn as nn
import numpy as np
import torch.autograd as autograd

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


def calc_gradient_penalty(netD, real_data, fake_data, device, LAMBDA):
    batchsize = real_data.shape[0]
    alpha = torch.rand(batchsize, 1) # sample from uniform [0,1)
    alpha = alpha.expand(batchsize, np.int(real_data.nelement()/batchsize)).view(batchsize, 1, 100, 150).to(device)
    
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(device)
    
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    
    disc_interpolates = netD(interpolates)[0]
    
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size(), device=device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty



class ShapeAutoencoder(nn.Module):
    def __init__(self, image_size, latent_size, hidden_size, device, LAMBDA):
        super(ShapeAutoencoder, self).__init__()
        self.image_size = image_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.device = device
        self.LAMBDA = LAMBDA
        self.decoder = Decoder(image_size, hidden_size, latent_size)
    
    def compute_loss_WGANGP(self, discriminator, optimizer, minibatch, rand_dist):
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        y_data = discriminator(data)
        y_fake = discriminator(data_fake)
        errD_real = torch.mean(y_data)
        errD_fake = torch.mean(y_fake)
        gradient_penalty = calc_gradient_penalty(discriminator, data, data_fake.view(-1,1,100,150), self.device, self.LAMBDA)
        errD = -errD_real + errD_fake + gradient_penalty
        optimizer.zero_grad()
        errD.backward()
        optimizer.step()
        
        return errD
    
    def compute_loss_G(self, discriminator, minibatch, rand_dist):
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        y_fake = discriminator(data_fake)
        errG = -torch.mean(y_fake)
        return errG

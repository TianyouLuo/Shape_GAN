import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

### For WGAN-GP, all batch norm layers need to be removed
class Discriminator(nn.Module):
    def __init__(self, image_size, latent_size, num_chanel, hidden_chanels=64):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        self.latent_size = latent_size
        self.num_chanel = num_chanel
        self.hidden_chanels = hidden_chanels
        self.main1 = nn.Sequential(
            # input is (nc) x 100 x 150
            nn.Conv2d(self.num_chanel, self.hidden_chanels, 6, 4, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 25 x 37
            nn.Conv2d(self.hidden_chanels, self.hidden_chanels * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(self.hidden_chanels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 12 x 18
            nn.Conv2d(self.hidden_chanels * 2, self.hidden_chanels * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(self.hidden_chanels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 6 x 9
            nn.Conv2d(self.hidden_chanels * 4, self.hidden_chanels * 8, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(self.hidden_chanels * 8),
            #nn.Tanh(),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.main2 = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.hidden_chanels * 8, 1, [3, 4], 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Sigmoid(),
        )

        self.mainz = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.latent_size, self.hidden_chanels * 8, 1, stride=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.hidden_chanels * 8, self.hidden_chanels * 8 * 4 * 4, 1, stride=1, bias=False),
            #nn.BatchNorm2d(self.hidden_chanels * 8 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        #self.fc = nn.Sequential(nn.Linear(self.hidden_chanels * 8 * 4 * 4 * 2, 1), nn.Sigmoid())
        self.fc = nn.Sequential(nn.Linear(self.hidden_chanels * 8 * 4 * 4 * 2, 1))

    def forward(self, x, z=None, flag=False):
        if flag is False:
            h = self.main1(x)
            y = self.main2(h).view(x.shape[0], -1)
        else:
            h = self.main1(x).view(x.shape[0], -1)
            h2 = self.mainz(z.view(z.shape[0], self.latent_size, 1, 1)).view(z.shape[0], -1)
            y = self.fc(torch.cat([h, h2], dim=1))
        return y, h


class LSUNDecoder(nn.Module):
    def __init__(self, image_size, latent_size, num_chanel, hidden_chanels=64):
        super(LSUNDecoder, self).__init__()
        self.image_size = image_size
        self.latent_size = latent_size
        self.num_chanel = num_chanel
        self.hidden_chanels = hidden_chanels

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.latent_size, self.hidden_chanels * 8, [4, 5], 1, 0, bias=False),
            nn.BatchNorm2d(self.hidden_chanels * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 5
            nn.ConvTranspose2d(self.hidden_chanels * 8, self.hidden_chanels * 4, [2, 3], 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_chanels * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 6 x 9
            nn.ConvTranspose2d(self.hidden_chanels * 4, self.hidden_chanels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_chanels * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 12 x 18
            nn.ConvTranspose2d(self.hidden_chanels * 2, self.hidden_chanels, 4, 2, 1, 1, bias=False),
            nn.BatchNorm2d(self.hidden_chanels),
            nn.ReLU(True),
            # state size. (ngf) x 25 x 37
            nn.ConvTranspose2d(self.hidden_chanels, self.num_chanel, 6, 4, 1, [0, 2], bias=False),
            nn.Tanh()
            # state size. (nc) x 100 x 150
        )

    def forward(self, z):
        x = self.main(z.view(z.shape[0], self.latent_size, 1, 1))
        return x


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




class DCGANAE(nn.Module):
    def __init__(self, image_size, latent_size, num_chanel, hidden_chanels, device, LAMBDA):
        super(DCGANAE, self).__init__()
        self.image_size = image_size
        self.num_chanel = num_chanel
        self.latent_size = latent_size
        self.hidden_chanels = hidden_chanels
        self.LAMBDA = LAMBDA
        self.device = device
        self.decoder = LSUNDecoder(image_size, latent_size, num_chanel, hidden_chanels)
    
    def compute_loss_WGANGP(self, discriminator, optimizer, minibatch, rand_dist):
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        y_data, _ = discriminator(data.detach())
        y_fake, _ = discriminator(data_fake.detach())
        errD_real = torch.mean(y_data)
        errD_fake = torch.mean(y_fake)
        gradient_penalty = calc_gradient_penalty(discriminator, data, data_fake, self.device, self.LAMBDA)
        errD = -errD_real + errD_fake + gradient_penalty
        optimizer.zero_grad()
        errD.backward()
        optimizer.step()
        
        return errD
    
    def compute_loss_G(self, discriminator, minibatch, rand_dist):
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        y_fake, _ = discriminator(data_fake.detach())
        errG = -torch.mean(y_fake)
        return errG
    





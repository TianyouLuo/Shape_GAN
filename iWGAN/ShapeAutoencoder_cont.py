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


class Encoder(nn.Module):
    def __init__(self, image_size, hidden_size, latent_size):
        super(Encoder, self).__init__()
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
            nn.Linear(hidden_size, self.latent_size, bias=False),
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


### MMD penalty ###
# MMD loss between z and Q(x)
def mmd_penalty(z_hat, z, device, kernel="RBF", sigma2_p=1):
    n = z.shape[0]
    zdim = z.shape[1]
    half_size = int((n * n - n)/2)
    
    norms_z = z.pow(2).sum(1).unsqueeze(1)
    dots_z = torch.mm(z, z.t())
    dists_z = (norms_z + norms_z.t() - 2. * dots_z).abs()
    
    norms_zh = z_hat.pow(2).sum(1).unsqueeze(1)
    dots_zh = torch.mm(z_hat, z_hat.t())
    dists_zh = (norms_zh + norms_zh.t() - 2. * dots_zh).abs()
    
    dots = torch.mm(z_hat, z.t())
    dists = (norms_zh + norms_z.t() - 2. * dots).abs()
    
    if kernel == "RBF":
        sigma2_k = torch.topk(dists.reshape(-1), half_size)[0][-1]
        sigma2_k = sigma2_k + torch.topk(dists_zh.reshape(-1), half_size)[0][-1]
        
        res1 = torch.exp(-dists_zh/2./sigma2_k)
        res1 = res1 + torch.exp(-dists_z/2./sigma2_k)
        res1 = torch.mul(res1, 1. - torch.eye(n, device = device))
        res1 = res1.sum() / (n*n-n)
        res2 = torch.exp(-dists/2./sigma2_k)
        res2 = res2.sum()*2./(n*n)
        stat = res1 - res2
        return stat
    
    elif kernel == "IMQ":
        Cbase = 2 * zdim * sigma2_p
        stat = 0.
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = Cbase * scale
            res1 = C / (C + dists_z) + C / (C + dists_zh)
            res1 = torch.mul(res1, 1. - torch.eye(n))
            res1 = res1.sum() / (n*n-n)
            res2 = C / (C + dists)
            res2 = res2.sum()*2./(n*n)
            stat = stat + res1 - res2
        return stat



class ShapeAutoencoder(nn.Module):
    def __init__(self, image_size, latent_size, hidden_size, device, lambda_gp, lambda_mmd, lambda_recon):
        super(ShapeAutoencoder, self).__init__()
        self.image_size = image_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.device = device
        self.lambda_gp = lambda_gp
        self.lambda_mmd = lambda_mmd
        self.lambda_recon = lambda_recon
        self.decoder = Decoder(image_size, hidden_size, latent_size)
        self.encoder = Encoder(image_size, hidden_size, latent_size)
    
    def compute_loss_WGANGP_orig(self, discriminator, optimizer, minibatch, rand_dist):
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        y_data = discriminator(data)
        y_fake = discriminator(data_fake)
        errD_real = torch.mean(y_data)
        errD_fake = torch.mean(y_fake)
        gradient_penalty = calc_gradient_penalty(discriminator, data, data_fake.view(-1,1,100,150), self.device, self.lambda_gp)
        errD = -errD_real + errD_fake + gradient_penalty
        optimizer.zero_grad()
        errD.backward()
        optimizer.step()
        
        return errD, gradient_penalty
    
    def compute_loss_G(self, discriminator, minibatch, rand_dist):
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        y_fake = discriminator(data_fake)
        errG = -torch.mean(y_fake)
        return errG
    
    def compute_loss_WGANGP(self, discriminator, optimizer, minibatch, rand_dist):
        data = minibatch.to(self.device)
        data_embed = self.encoder(data)
        data_reconst = self.decoder(data_embed)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        y_data = discriminator(data_reconst)
        y_fake = discriminator(data_fake)
        errD_real = torch.mean(y_data)
        errD_fake = torch.mean(y_fake)
        gradient_penalty = calc_gradient_penalty(discriminator, data, data_fake.view(-1,1,100,150), self.device, self.lambda_gp)
        errD = -errD_real + errD_fake + gradient_penalty
        optimizer.zero_grad()
        errD.backward()
        optimizer.step()
        
        return errD, gradient_penalty
    
    def compute_loss_GQ(self, discriminator, minibatch, rand_dist):
        data = minibatch.to(self.device)
        z_hat = self.encoder(data)
        data_reconst = self.decoder(z_hat)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        y_data = discriminator(data_reconst)
        y_fake = discriminator(data_fake)
        errG_real = torch.mean(y_data)
        errG_fake = torch.mean(y_fake)
        dim = list(range(1, 4))
        l2 = torch.mean(torch.sqrt(torch.sum((data-data_reconst.view(-1,1,100,150))**2, dim=dim)))
        GQ_cost = errG_real - errG_fake + self.lambda_recon * l2
        mmd = mmd_penalty(z_hat, z_prior, self.device, kernel="RBF")
        primal_cost = GQ_cost + self.lambda_mmd * mmd
        return primal_cost, mmd, l2

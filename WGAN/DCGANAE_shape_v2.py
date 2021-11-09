import torch
import torch.nn as nn


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
            nn.BatchNorm2d(self.hidden_chanels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 12 x 18
            nn.Conv2d(self.hidden_chanels * 2, self.hidden_chanels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_chanels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 6 x 9
            nn.Conv2d(self.hidden_chanels * 4, self.hidden_chanels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hidden_chanels * 8),
            #nn.Tanh(),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.main2 = nn.Sequential(
            # state size. (ndf*8) x 3 x 4
            #nn.Linear(self.hidden_chanels * 8 * 3 * 4, 1)
            nn.Conv2d(self.hidden_chanels * 8, 1, [3, 4], 1, 0, bias=False),
            nn.Sigmoid(),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Sigmoid(),
        )
        
        self.mainz = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.latent_size, self.hidden_chanels * 8, 1, stride=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.hidden_chanels * 8, self.hidden_chanels * 8 * 4 * 4, 1, stride=1, bias=False),
            nn.BatchNorm2d(self.hidden_chanels * 8 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        #self.fc = nn.Sequential(nn.Linear(self.hidden_chanels * 8 * 4 * 4 * 2, 1), nn.Sigmoid())
        self.fc = nn.Sequential(nn.Linear(self.hidden_chanels * 8 * 4 * 4 * 2, 1))
    
    def forward(self, x, z=None, flag=False):
        if flag is False:
            h = self.main1(x)
            #y = self.main2(h.view(x.shape[0],self.hidden_chanels * 8 * 3 * 4)).view(x.shape[0], -1)
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


def wasserstein_loss(first_samples, second_samples, max_iter=10, lam=1, device="cuda"):
    embedding_dim = first_samples.size(1)
    pro = rand_projections(embedding_dim, num_projections).to(device)
    first_samples_detach = first_samples.detach()
    second_samples_detach = second_samples.detach()
    for _ in range(max_iter):
        projections = f(pro)
        cos = cosine_distance_torch(projections, projections)
        reg = lam * cos
        encoded_projections = first_samples_detach.matmul(projections.transpose(0, 1))
        distribution_projections = second_samples_detach.matmul(projections.transpose(0, 1))
        wasserstein_distance = torch.abs(
            (
                torch.sort(encoded_projections.transpose(0, 1), dim=1)[0]
                - torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]
            )
        )
        wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1), 1.0 / p)
        wasserstein_distance = torch.pow(torch.pow(wasserstein_distance, p).mean(), 1.0 / p)
        loss = reg - wasserstein_distance
        f_op.zero_grad()
        loss.backward(retain_graph=True)
        f_op.step()

    projections = f(pro)
    encoded_projections = first_samples.matmul(projections.transpose(0, 1))
    distribution_projections = second_samples.matmul(projections.transpose(0, 1))
    wasserstein_distance = torch.abs(
        (
            torch.sort(encoded_projections.transpose(0, 1), dim=1)[0]
            - torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]
        )
    )
    wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1), 1.0 / p)
    wasserstein_distance = torch.pow(torch.pow(wasserstein_distance, p).mean(), 1.0 / p)
    return wasserstein_distance



class DCGANAE(nn.Module):
    def __init__(self, image_size, latent_size, num_chanel, hidden_chanels, device):
        super(DCGANAE, self).__init__()
        self.image_size = image_size
        self.num_chanel = num_chanel
        self.latent_size = latent_size
        self.hidden_chanels = hidden_chanels
        self.device = device
        self.decoder = LSUNDecoder(image_size, latent_size, num_chanel, hidden_chanels)
    
    def compute_loss_WGAN(self, discriminator, optimizer, minibatch, rand_dist, tnet, op_tnet, p=2, max_iter=100, lam=1):
        label = torch.full((minibatch.shape[0], 1), 1, dtype=torch.float32, device=self.device)
        criterion = nn.BCELoss()
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        y_data, _ = discriminator(data.detach())
        errD0_real = criterion(y_data, label)
        optimizer.zero_grad()
        errD0_real.backward()
        optimizer.step()
        y_fake, _ = discriminator(data_fake.detach())
        label.fill_(0)
        errD0_fake = criterion(y_fake, label)
        optimizer.zero_grad()
        errD0_fake.backward()
        optimizer.step()
        
        _, data = discriminator(data)
        _, data_fake = discriminator(data_fake)
        _wd = wasserstein_loss(
            data.view(data.shape[0], -1),
            data_fake.view(data.shape[0], -1),
            tnet,
            op_tnet,
            max_iter,
            lam,
            self.device,
        )
        errD_real = torch.mean(y_data)
        errD_fake = torch.mean(y_fake)
        errD = -errD_real + errD_fake
        optimizer.zero_grad()
        errD.backward()
        optimizer.step()
        
        return _wd
    
    def compute_loss_G(self, discriminator, minibatch, rand_dist):
        data = minibatch.to(self.device)
        z_prior = rand_dist((data.shape[0], self.latent_size)).to(self.device)
        data_fake = self.decoder(z_prior)
        y_fake, _ = discriminator(data_fake.detach())
        errG = -torch.mean(y_fake)
        return errG
    





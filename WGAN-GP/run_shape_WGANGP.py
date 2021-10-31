from __future__ import print_function

import argparse
import os

import numpy as np
import glob
import torch
#import torchvision.datasets as datasets
from DCGANAE_shape import DCGANAE, Discriminator
#from experiments import reconstruct, sampling, sampling_eps, sampling_shape
import pandas as pd
# torch.backends.cudnn.enabled = False
# from ShapeAutoencoder import ShapeAutoencoder
from torch import optim
#from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
#from TransformNet import TransformNet
#from utils import circular_function, compute_true_Wasserstein, save_dmodel, sliced_wasserstein_distance


# train args
parser = argparse.ArgumentParser(description="Disributional Sliced Wasserstein Autoencoder")
parser.add_argument("--datadir", default="./", help="path to dataset")
parser.add_argument("--outdir", default="./result", help="directory to output images")
parser.add_argument(
    "--batch-size", type=int, default=512, metavar="N", help="input batch size for training (default: 512)"
)
parser.add_argument(
    "--epochs", type=int, default=200, metavar="N", help="number of epochs to train (default: 200)"
)
parser.add_argument("--lr", type=float, default=0.0005, metavar="LR", help="learning rate (default: 0.0005)")
parser.add_argument(
    "--num-workers",
    type=int,
    default=16,
    metavar="N",
    help="number of dataloader workers if device is CPU (default: 16)",
)
parser.add_argument("--seed", type=int, default=16, metavar="S", help="random seed (default: 16)")
parser.add_argument("--g", type=str, default="circular", help="g")
parser.add_argument("--num-projection", type=int, default=1000, help="number projection")
parser.add_argument("--lam", type=float, default=1, help="Regularization strength")
parser.add_argument("--p", type=int, default=2, help="Norm p")
parser.add_argument("--d_iter", type=int, default=10, help="number of discriminator iterations")
#parser.add_argument("--clip", type=float, default=0.01, help="clip value")
parser.add_argument("--r", type=float, default=1000, help="R")
parser.add_argument("--kappa", type=float, default=50, help="R")
parser.add_argument("--k", type=int, default=10, help="R")
parser.add_argument("--LAMBDA", type=float, default=10, help="lambda for gradient penalty")
parser.add_argument("--latent-size", type=int, default=32, help="Latent size")
parser.add_argument("--hsize", type=int, default=100, help="h size")
parser.add_argument("--dataset", type=str, default="MNIST", help="(MNIST|FMNIST)")
parser.add_argument(
    "--model-type", type=str, required=True, default="WGANGP"
)
args = parser.parse_args()

torch.random.manual_seed(args.seed)
#if args.g == "circular":
#    g_function = circular_function
model_type = args.model_type
latent_size = args.latent_size
dataset = args.dataset
image_size = 100 * 150
num_chanel = 1
d_iter = args.d_iter
#clip_value = args.clip

if model_type == "WGANGP":
    model_dir = os.path.join(
            args.outdir, model_type + "_iter" + str(d_iter) + "_lr" + str(args.lr) + "_penal" + str(args.LAMBDA)
            )


save_dir = os.path.join(model_dir, "generated_images")
if not (os.path.isdir(args.datadir)):
    os.makedirs(args.datadir)
if not (os.path.isdir(args.outdir)):
    os.makedirs(args.outdir)
if not (os.path.isdir(model_dir)):
    os.makedirs(model_dir)
if not (os.path.isdir(save_dir)):
    os.makedirs(save_dir)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(
    "batch size {}\nepochs {}\nAdam lr {} \n using device {}\n".format(
        args.batch_size, args.epochs, args.lr, device.type
    )
)
# build training set data loaders
if(dataset=='Yalin'):
    datadir='/pine/scr/t/i/tianyou/Yalin_GAN/data/ADNIHippoCSV/'
    files = glob.glob(datadir+'/*.csv')
    
    train_data = np.zeros((len(files),1,100,150), dtype=np.float32)
    
    # checking all the csv files in the 
    # specified path
    for l in range(len(files)):
        # reading content of csv file
        # content.append(filename)
        if l % 50 == 0:
            print(len(files) - l)
        filename = files[l]
        df = pd.read_csv(filename, names=['f1','f2','f3','f4','f5','f6','f7'],sep=' ')
        df_sel = df.iloc[0:15000,0].to_numpy(dtype = np.float32)  ## only work with one modality for now
        df_reshape = df_sel.reshape((100,150), order = 'F')
        train_data[l,0,:,:] = df_reshape
    
    pmax = np.max(train_data)
    pmin = np.min(train_data)
    train_norm = train_data / pmax
    X = torch.tensor(train_norm, dtype=torch.float32).to(device)
    train_loader = torch.utils.data.DataLoader(X, batch_size=args.batch_size, shuffle=True)
    model = DCGANAE(image_size, latent_size=latent_size, num_chanel=1, hidden_chanels=64, device=device,
                    LAMBDA = args.LAMBDA).to(device)
    dis = Discriminator(image_size, latent_size, 1, 64).to(device)
    disoptimizer = optim.Adam(dis.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
#if model_type == "WGAN":
#    # Dimension of transform_net is hidden_channels * 8 * dim of last layer in discriminator (h)
#    transform_net = TransformNet(64 * 8 * 3 * 4).to(device)
#    op_trannet = optim.Adam(transform_net.parameters(), lr=args.lr, betas=(0.5, 0.999))
    # op_trannet = optim.Adam(transform_net.parameters(), lr=1e-4)
    # train_net(28 * 28, 1000, transform_net, op_trannet)



optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
fixednoise = torch.randn((16, latent_size)).to(device)
finalnoise = torch.randn((256, latent_size)).to(device)
ite = 0
wd_list = []
swd_list = []
for epoch in range(args.epochs):
    total_loss = 0.0
    data_iter = iter(train_loader)
    i = 0
    while i < len(train_loader):
    # For each batch in the dataloader
        ## update D network
        j = 0
        critic_iter = d_iter
        
        while j < critic_iter and i < len(train_loader):
            j += 1
            data = data_iter.next()
            i += 1
            loss = model.compute_loss_WGANGP(dis, disoptimizer, data, torch.randn)
            
            #for p in dis.parameters():
            #    p.data = torch.clamp(p.data, -clip_value, clip_value)
            
        ############################
        # (2) Update G network
        ###########################
        #data = data_iter.next()
        #i += 1
        lossG = model.compute_loss_G(dis, data, torch.randn)
        optimizer.zero_grad()
        lossG.backward()
        optimizer.step()
        total_loss += loss.item()
        ite += 1
        
#        if ite % 100 == 0:
#            model.eval()
#            for _, (input, y) in enumerate(test_loader, start=0):
#                fixednoise_wd = torch.randn((10000, latent_size)).to(device)
#                data = input.to(device)
#                data = data.view(data.shape[0], -1)
#                fake = model.decoder(fixednoise_wd)
#                wd_list.append(compute_true_Wasserstein(data.to("cpu"), fake.to("cpu")))
#                swd_list.append(sliced_wasserstein_distance(data, fake, 10000).item())
#                print("Iter:" + str(ite) + " WD: " + str(wd_list[-1]))
#                np.savetxt(model_dir + "/wd.csv", wd_list, delimiter=",")
#                print("Iter:" + str(ite) + " SWD: " + str(swd_list[-1]))
#                np.savetxt(model_dir + "/swd.csv", swd_list, delimiter=",")
#                break
#            model.train()
    total_loss /= ite + 1
    print("Epoch: " + str(epoch) + " Loss: " + str(total_loss))
    
#    if epoch % 10 == 0:
#        model.eval()
#        sampling_shape(
#            model_dir + "/sample_epoch_" + str(epoch) + ".png",
#            fixednoise,
#            model.decoder,
#            pmax,
#            16,
#            num_chanel,
#        )
#        if model_type[0] == "J":
#            for _, (input, y) in enumerate(test_loader2, start=0):
#                input = input.to(device)
#                input = input.view(-1, image_size ** 2)
#                reconstruct(
#                    model_dir + "/reconstruction_epoch_" + str(epoch) + ".png",
#                    input,
#                    model.encoder,
#                    model.decoder,
#                    image_size,
#                    num_chanel,
#                    device,
#                )
#                break
#        model.train()
    #save_dmodel(model, optimizer, None, None, None, None, epoch, model_dir)
    if (epoch % 40 == 0):
        model.eval()
        samp = model.decoder(fixednoise).view(-1, 100 * 150)
        samp_num = samp.detach().to('cpu').numpy()
        samp_pd = pd.DataFrame(samp_num * pmax)
        samp_pd.to_csv(model_dir+"/samp_epoch"+str(epoch)+".csv", index = False)
        model.train()
        
#    if epoch == args.epochs - 1:
#        model.eval()
#        sampling_eps(
#            model_dir + "/sample_epoch_" + str(epoch), fixednoise, model.decoder, 64, image_size, num_chanel
#        )
#        model.train()

# store the final samples
#model.eval()
#samp_final = model.decoder(finalnoise)
#samp_final_num = samp_final.detach().to('cpu').numpy()
#for img in range(samp_final.shape[0]):
#    save_image(samp_final[img,:,:,:] * pmax, save_dir + "/img_" + str(img) + ".png", normalize=False)
#    samp_final_pd = pd.DataFrame(samp_final_num[img,0,:,:] * pmax)
#    samp_final_pd.to_csv(save_dir+"/img"+str(img)+".csv", index = False)

# Only need to run one time: save original images as npy
#for img in range(train_data.shape[0]):
#    save_image(X[img,:,:,:] * pmax, "/pine/scr/t/i/tianyou/Yalin_GAN/data/train_npy" + "/img_" + str(img) + ".png", normalize=False)
#    train_pd = pd.DataFrame(train_data[img,0,:,:])
#    train_pd.to_csv("/pine/scr/t/i/tianyou/Yalin_GAN/data/train_npy"+"/img"+str(img)+".csv", index = False)

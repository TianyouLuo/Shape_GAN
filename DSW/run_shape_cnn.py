from __future__ import print_function

import argparse
import os

import numpy as np
import glob
import torch
import torchvision.datasets as datasets
from DCGANAE_shape import DCGANAE, Discriminator
from experiments import reconstruct, sampling, sampling_eps, sampling_shape
from gsw import GSW
import pandas as pd
# torch.backends.cudnn.enabled = False
from gswnn import GSW_NN
from ShapeAutoencoder import ShapeAutoencoder
from torch import optim
from torchvision import transforms
from tqdm import tqdm
from TransformNet import TransformNet
from utils import circular_function, compute_true_Wasserstein, save_dmodel, sliced_wasserstein_distance


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
parser.add_argument("--niter", type=int, default=10, help="number of iterations")
parser.add_argument("--r", type=float, default=1000, help="R")
parser.add_argument("--kappa", type=float, default=50, help="R")
parser.add_argument("--k", type=int, default=10, help="R")
parser.add_argument("--e", type=float, default=1000, help="R")
parser.add_argument("--latent-size", type=int, default=32, help="Latent size")
parser.add_argument("--hsize", type=int, default=100, help="h size")
parser.add_argument("--dataset", type=str, default="MNIST", help="(MNIST|FMNIST)")
parser.add_argument(
    "--model-type", type=str, required=True, help="(SWD|MSWD|DSWD|GSWD|DGSWD|JSWD|JMSWD|JDSWD|JGSWD|JDGSWD)"
)
args = parser.parse_args()

torch.random.manual_seed(args.seed)
if args.g == "circular":
    g_function = circular_function
model_type = args.model_type
latent_size = args.latent_size
num_projection = args.num_projection
dataset = args.dataset
image_size = 100 * 150
num_chanel = 1
assert model_type in [
    "MGSWNN",
    "JMGSWNN",
    "SWD",
    "MSWD",
    "MGSWD",
    "DSWD",
    "GSWD",
    "DGSWD",
    "JSWD",
    "JMSWD",
    "JMGSWD",
    "JDSWD",
    "JGSWD",
    "JDGSWD",
    "DGSWNN",
    "JDGSWNN",
    "GSWNN",
    "JGSWNN",
]
if model_type == "SWD" or model_type == "JSWD":
    model_dir = os.path.join(args.outdir, model_type + "_n" + str(num_projection))
elif model_type == "GSWD" or model_type == "JGSWD":
    model_dir = os.path.join(args.outdir, model_type + "_n" + str(num_projection) + "_" + args.g + str(args.r))
elif model_type == "DSWD" or model_type == "JDSWD":
    model_dir = os.path.join(
        args.outdir, model_type + "_iter" + str(args.niter) + "_n" + str(num_projection) + "_lam" + str(args.lam)
    )
elif model_type == "DGSWD" or model_type == "JDGSWD":
    model_dir = os.path.join(
        args.outdir,
        model_type
        + "_iter"
        + str(args.niter)
        + "_n"
        + str(num_projection)
        + "_lam"
        + str(args.lam)
        + "_"
        + args.g
        + str(args.r),
    )
elif model_type == "MSWD" or model_type == "JMSWD":
    model_dir = os.path.join(args.outdir, model_type)
elif model_type == "MGSWNN" or model_type == "JMGSWNN":
    model_dir = os.path.join(args.outdir, model_type + "_size" + str(args.hsize))
elif model_type == "MGSWD" or model_type == "JMGSWD":
    model_dir = os.path.join(args.outdir, model_type + "_" + args.g)
if not (os.path.isdir(args.datadir)):
    os.makedirs(args.datadir)
if not (os.path.isdir(args.outdir)):
    os.makedirs(args.outdir)
if not (os.path.isdir(args.outdir)):
    os.makedirs(args.outdir)
if not (os.path.isdir(model_dir)):
    os.makedirs(model_dir)
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
    model = DCGANAE(image_size, latent_size=latent_size, num_chanel=1, hidden_chanels=64, device=device).to(device)
    dis = Discriminator(image_size, latent_size, 1, 64).to(device)
    disoptimizer = optim.Adam(dis.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
if model_type == "DSWD" or model_type == "DGSWD":
    # Dimension of transform_net is hidden_channels * 8 * dim of last layer in discriminator (h)
    transform_net = TransformNet(64 * 8 * 3 * 4).to(device)
    op_trannet = optim.Adam(transform_net.parameters(), lr=args.lr, betas=(0.5, 0.999))
    # op_trannet = optim.Adam(transform_net.parameters(), lr=1e-4)
    # train_net(28 * 28, 1000, transform_net, op_trannet)
elif model_type == "JDSWD" or model_type == "JDSWD2" or model_type == "JDGSWD":
    transform_net = TransformNet(args.latent_size + 28 * 28).to(device)
    op_trannet = optim.Adam(transform_net.parameters(), lr=args.lr, betas=(0.5, 0.999))
    # train_net(args.latent_size + 28 * 28, 1000, transform_net, op_trannet)
if model_type == "MGSWNN":
    gsw = GSW_NN(din=28 * 28, nofprojections=1, model_depth=3, num_filters=args.hsize, use_cuda=True)
if model_type == "GSWNN" or model_type == "DGSWNN":
    gsw = GSW_NN(din=28 * 28, nofprojections=num_projection, model_depth=3, num_filters=args.hsize, use_cuda=True)
if model_type == "JMGSWNN":
    gsw = GSW_NN(din=28 * 28 + 32, nofprojections=1, model_depth=3, num_filters=args.hsize, use_cuda=True)
if model_type == "MSWD" or model_type == "JMSWD":
    gsw = GSW()
if model_type == "MGSWD":
    theta = torch.randn((1, 784), device=device, requires_grad=True)
    theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1))
    opt_theta = optim.Adam(transform_net.parameters(), lr=args.lr, betas=(0.5, 0.999))
if model_type == "JMGSWD":
    theta = torch.randn((1, 784 + 32), device=device, requires_grad=True)
    theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1))
    opt_theta = torch.optim.Adam(transform_net.parameters(), lr=args.lr, betas=(0.5, 0.999))
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
fixednoise = torch.randn((16, latent_size)).to(device)
ite = 0
wd_list = []
swd_list = []
for epoch in range(args.epochs):
    total_loss = 0.0
    for batch_idx, data in tqdm(enumerate(train_loader, start=0)):
        if model_type == "SWD":
            loss = model.compute_loss_SWD(dis, disoptimizer, data, torch.randn, num_projection, p=args.p)
        elif model_type == "GSWD":
            loss = model.compute_loss_GSWD(
                    dis, disoptimizer, data, torch.randn, g_function, args.r, num_projection, p=args.p)
        elif model_type == "MGSWNN":
            loss = model.compute_loss_MGSWNN(dis, disoptimizer, data, torch.randn, gsw, p=args.p)
        elif model_type == "MSWD":
            loss = model.compute_loss_MSWD(dis, disoptimizer, data, torch.randn, gsw)
        elif model_type == "DSWD":
            loss = model.compute_lossDSWD(
                    dis,
                    disoptimizer,
                    data,
                    torch.randn,
                    num_projection,
                    transform_net,
                    op_trannet,
                    p=args.p,
                    max_iter=args.niter,
                    lam=args.lam,
                    )
        elif model_type == "DGSWD":
            loss = model.compute_lossDGSWD(
                    dis,
                    disoptimizer,
                    data,
                    torch.randn,
                    num_projection,
                    transform_net,
                    op_trannet,
                    g_function,
                    args.r,
                    p=args.p,
                    max_iter=args.niter,
                    lam=args.lam,
                    )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
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
        ite = ite + 1
    total_loss /= batch_idx + 1
    print("Epoch: " + str(epoch) + " Loss: " + str(total_loss))
    
    if epoch % 10 == 0:
        model.eval()
        sampling_shape(
            model_dir + "/sample_epoch_" + str(epoch) + ".png",
            fixednoise,
            model.decoder,
            pmax,
            16,
            num_chanel,
        )
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
    save_dmodel(model, optimizer, None, None, None, None, epoch, model_dir)
    if (epoch % 10 == 0):
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



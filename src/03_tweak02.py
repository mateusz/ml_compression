#%%

from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torchvision.transforms as transforms
from torchvision.utils import save_image
import math
import matplotlib.pyplot as plt
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN
import os
import src.lib as slib

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
name = '03f'

"""
This is an update from 02, with a reliable result (because I actually compress
and decompress the image to verify the result).

This notebook outputs an uncompressed parquet file containing the compressed
data, plus a PNG image for display.

The actual metrics are computed in 99_jpeg.
"""

#%%
class ImgSet(Dataset):
    def __init__(self, path, tile=256, wtiles=-1, htiles=-1):
        super().__init__()

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        w,h=image.shape[0:2]
        padw = math.ceil(w/tile)*tile-w
        padh = math.ceil(h/tile)*tile-h
        image = cv2.copyMakeBorder(image, 0, padw, 0, padh, cv2.BORDER_CONSTANT, value=[0,0,0])

        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img = transform(image)
        # tile image
        img = img.unfold(1, tile, tile).unfold(2, tile, tile)
        self.htiles = img.shape[1]
        self.wtiles = img.shape[2]

        if wtiles>0:
            self.wtiles = wtiles
        if htiles>0:
            self.htiles = htiles
        self.img = img[:,:htiles,:wtiles].flatten(1,2).permute(1,0,2,3).to(device)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        return self.img[idx]

tile=128
wtiles=32
htiles=32
batch_size=16
# 256 tiles (with batch 8) converged
iset = ImgSet('data/STScI-01GA76Q01D09HFEV174SVMQDMV.png', tile=tile, wtiles=wtiles, htiles=htiles)
print(iset.img.shape, iset.wtiles, iset.htiles)

train = DataLoader(iset, batch_size=batch_size, shuffle=True)
print(len(train))

#%%

class Network(nn.Module):
    def __init__(self, N=128):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.encode = nn.Sequential(
            nn.Conv2d(3, N, stride=2, kernel_size=5, padding=2),
            #nn.ReLU(inplace=True),
            GDN(N),
            nn.Conv2d(N, N, stride=2, kernel_size=5, padding=2),
            GDN(N),
            #nn.ReLU(inplace=True),
            nn.Conv2d(N, N, stride=2, kernel_size=5, padding=2),
        )

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(N, N, kernel_size=5, padding=2, output_padding=1, stride=2),
            GDN(N, inverse=True),
            #nn.ReLU(inplace=True),
            nn.ConvTranspose2d(N, N, kernel_size=5, padding=2, output_padding=1, stride=2),
            #nn.ReLU(inplace=True),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, 3, kernel_size=5, padding=2, output_padding=1, stride=2),
        )

    def forward(self, x):
       y = self.encode(x)
       y_hat, y_likelihoods = self.entropy_bottleneck(y)
       x_hat = self.decode(y_hat)
       return x_hat, y_likelihoods

net = Network(N=32).to(device)
print(net(iset[0].unsqueeze(0))[0].shape)
print(sum([l.numel() for l in net.parameters()]))
min_loss = 9999999

def show_img(iset, data, size=5):
    h = iset.htiles*tile
    w = iset.wtiles*tile
    # Reshape into m x n
    merged = data.view(iset.htiles,iset.wtiles,3,tile,tile).permute(2,0,3,1,4)
    # Merge tiles
    merged = merged.reshape(3, h, w)
    # Prep for pyplot
    merged = merged.detach().cpu().permute(1,2,0)

    fig,ax = plt.subplots(figsize=(size,size))
    ax.axis('off')
    ax.imshow(merged)
    plt.show()

def show_tile(tile, tile2, size=5):
    # Prep for pyplot
    tile = tile.permute(1,2,0).clamp(0.0,1.0).detach().cpu()
    tile2 = tile2.permute(1,2,0).clamp(0.0,1.0).detach().cpu()

    fig,ax = plt.subplots(1,2,figsize=(size*2,size))
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].imshow(tile)
    ax[1].imshow(tile2)
    plt.show()

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

# Note the models before 03f were using parquet compression on an unoptimised aux loss.
# The result seems good specially for 03c, but would need to investigated further
# because it can contain bugs :)
#%%

opt = torch.optim.Adam((net.encode+net.decode).parameters(), lr=0.001)
auxopt = torch.optim.Adam(net.entropy_bottleneck.parameters(), lr=0.01)


# min 161x161
# more border artifacts, more noisy stars
msssim = MS_SSIM(data_range=1, size_average=True, channel=3)

# Produces better results after half the epochs, with higher bpp (1.7 vs 0.5).
# Has edge artifacts, but reproduces better detail (e.g. webb artifacts)
# seems the best
# Good loss here is around 0.02-0.03
# One of the earlier papers (end to end?) mentions ssim/ms-sim can struggle with color pictures
# But actually both SSIM and MSE produce cyan-colored stars here.
ssim = SSIM(data_range=1, size_average=True, channel=3)
# maybe try mse+ssim?

# Usual smoothed version. Does not have webb artifacts.
# Good loss here is around 0.0002
mse = torch.nn.MSELoss()

perc_w = 1.0
# for mse, 0.0001 produces 0.5bpp
# for ssim, 0.0001 produces 1.87bpp
# for msssimg, 0.001 produces 1.33bpp
# for ssim, 0.01 produces 0.5bpp
bpp_w = 0.0007

# ssim works really well
# but need to solve tiling issue (overlap?)

if not os.path.exists('out/train%s' % (name)):
    os.mkdir('out/train%s' % (name))
net = net.to(device)
net.train()
early_stop = 0
report_steps = 1
image_steps = 1
show_steps = 1000000

try:
    for epoch in range(999999):
        running_loss = 0.0
        running_perc = 0.0
        running_bpp = 0.0
        running_auxloss = 0.0

        for i,t in enumerate(train,0):
            opt.zero_grad()
            auxopt.zero_grad()

            x_hat, y_likelihoods = net(t)

            perc_loss = 1.0 - ssim(t, x_hat)
            #perc_loss = 1.0 - msssim(t, x_hat)
            #perc_loss = mse(t, x_hat)

            N, _, H, W = t.size()
            num_pixels = N * H * W
            bpp_loss = torch.log(y_likelihoods).sum() / (-math.log(2) * num_pixels)

            loss = perc_w*perc_loss + bpp_w*bpp_loss

            loss.backward()
            # The optimisation likes to explode
            clip_gradient(opt, 1)
            opt.step()

            auxloss = net.entropy_bottleneck.loss()
            auxloss.backward()
            auxopt.step()

            running_loss += loss.item()
            running_bpp += bpp_loss.item()
            running_perc += perc_loss.item()
            running_auxloss += auxloss.item()

        running_loss /= len(train)
        running_bpp /= len(train)
        running_perc /= len(train)
        running_auxloss /= len(train)

        if epoch%report_steps==(report_steps-1):
            print("[%d] l=%.8f, bpp=%.4f, perc=%.8f, aux=%.4f (min=%.8f)" % (epoch, running_loss, running_bpp, running_perc, running_auxloss, min_loss))
        if epoch%image_steps==(image_steps-1):
            net.eval()
            res = net(iset[1].unsqueeze(0))[0].squeeze(0)
            #if epoch%show_steps==0:
                #show_tile(iset[1], net(iset[1].unsqueeze(0))[0].squeeze(0), size=5)
            save_image(res, 'out/train%s/%d.png' % (name, epoch))
            net.train()

        #if running_perc<0.032 and running_bpp<0.55:
        #    print('good enough')
        #    break

        if running_loss<min_loss:
            early_stop = 0
            min_loss = running_loss
            torch.save(net, 'models/%s' % name)
        else:
            early_stop += 1

        if early_stop>5:
            break

        running_loss = 0.0
        running_bpp = 0.0
        running_perc = 0.0
        running_auxloss = 0.0
except KeyboardInterrupt:
    print('interrupted!')

#%%

min_loss = 9999999
net = torch.load('models/%s' % name)
net = net.to(device)
net.train()
early_stop = 0
auxopt = torch.optim.Adam(net.entropy_bottleneck.parameters(), lr=0.01)
try:
    for epoch in range(999999):
        running_loss = 0.0

        for i,t in enumerate(train,0):
            auxopt.zero_grad()

            x_hat, y_likelihoods = net(t)
            auxloss = net.entropy_bottleneck.loss()
            auxloss.backward()
            auxopt.step()

            running_loss += auxloss.item()

        running_loss /= len(train)

        if epoch%report_steps==(report_steps-1):
            print("[%d] aux=%.4f (min=%.4f)" % (epoch, running_loss, min_loss))

        if running_loss<min_loss:
            early_stop = 0
            min_loss = running_loss
            torch.save(net, 'models/%s' % name)
        else:
            early_stop += 1

        if early_stop>3:
            break

        running_loss = 0.0
except KeyboardInterrupt:
    print('interrupted!')

#%%
net = torch.load('models/%s' % name)
net.eval()
o = None
collect = None
collect_out = None
torch.cuda.empty_cache()

test = DataLoader(iset, batch_size=batch_size, shuffle=False)
#%%

# Storage via writing tiles as byte strings into parquet rows.
shape = None
net.entropy_bottleneck.update()
collect = pd.DataFrame()
collect2 = pd.DataFrame()
for i,t in enumerate(test,0):
    #o = net(t)[0]
    #collect = torch.concat([collect,o.detach().cpu().clip(0.0,1.0)])

    # From https://github.com/InterDigitalInc/CompressAI/blob/master/compressai/models/base.py
    e = net.encode(t)
    bs = net.entropy_bottleneck.compress(e)
    if shape==None:
        shape = e.size()[2:]
    for b in bs:
        collect = collect.append({'chunk': b}, ignore_index=True)
collect.to_parquet('out/test-%s.parquet' % name, compression=None)
#%%

collect_out = slib.decompress_file(net, 'out/test-%s.parquet' % name, shape, batch_size)
h = iset.htiles*tile
w = iset.wtiles*tile
merged = collect_out.view(iset.htiles,iset.wtiles,3,tile,tile).permute(2,0,3,1,4)
merged = merged.reshape(3, h, w)
save_image(merged, 'out/test-%s.png' % name)

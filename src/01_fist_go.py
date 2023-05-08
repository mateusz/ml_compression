#%%

from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torchvision.transforms as transforms
import math
import matplotlib.pyplot as plt
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
01 is an early attempt to use a trivial autoencoder to reduce image's
dimensionality.  There is no image preprocessing done and no conversion of color
spaces.  I largely abandoned this approach after reading more about the
necessity to reduce entropy, and discovering that indeed the latent
representations produced here are just too large, and that AE isn't fit for this
purpose (essentially, dimensionality reduction != entropy reduction, these are
different problems)
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
wtiles=-1
htiles=-1
batch_size=128
# 256 tiles (with batch 8) converged
iset = ImgSet('data/STScI-01GA76Q01D09HFEV174SVMQDMV.png', tile=tile, wtiles=wtiles, htiles=htiles)
print(iset.img.shape, iset.wtiles, iset.htiles)

train = DataLoader(iset, batch_size=batch_size, shuffle=True)
print(len(train))

#%%
ch = 16
#latent = 512
net = nn.Sequential(
        nn.Conv2d(3, ch, 5),
        nn.MaxPool2d(2),
        #GDN(ch, device),
        nn.ReLU(inplace=True),
        #torch.nn.GroupNorm(1,ch),
        #nn.BatchNorm2d(ch),
        
        nn.Conv2d(ch, ch*2, 5),
        nn.MaxPool2d(2),
        #GDN(ch*2, device),
        nn.ReLU(inplace=True),
        #torch.nn.GroupNorm(1,ch*2),
        #nn.BatchNorm2d(ch*2),

        nn.Conv2d(ch*2, ch*4, 5),
        nn.MaxPool2d(2),
        nn.ReLU(inplace=True),
        #torch.nn.GroupNorm(1,ch*4),
        #nn.BatchNorm2d(ch*4),

        nn.Conv2d(ch*4, ch*8, 3),
        nn.MaxPool2d(2),
        nn.ReLU(inplace=True),
        #torch.nn.GroupNorm(1,ch*8),
        #nn.BatchNorm2d(ch*8),

        nn.Conv2d(ch*8, ch*16, 3),
        nn.MaxPool2d(2),
        nn.ReLU(inplace=True),
        #torch.nn.GroupNorm(1,ch*16),
        #nn.BatchNorm2d(ch*16),
        # 128,1,1

        #nn.Conv2d(ch*16, ch*64, 1),
        #nn.MaxPool2d(2),
        #nn.ReLU(inplace=True),
        #torch.nn.GroupNorm(1,ch*32),
        #nn.BatchNorm2d(ch*32),
        # 512,2,2

        #nn.Conv2d(ch*32, ch*64, 1),
        #nn.MaxPool2d(2),
        #nn.ReLU(inplace=True),
        # 1024,1,1

        #nn.Conv2d(ch*32, ch*32, 1),
        #nn.ReLU(inplace=True),

        nn.Flatten(),
        nn.Linear(ch*16, ch*16),
        nn.ReLU(),
        nn.Unflatten(1, (256,1,1)),
        
        #nn.Flatten(),
        #nn.Linear(ch*64, latent),
        #nn.ReLU(),
        #nn.Linear(latent, ch*64),
        #nn.ReLU(),
        #nn.Unflatten(1, (ch*64,1,1)),

        #nn.ConvTranspose2d(ch*64, ch*32, 2, stride=1),
        #nn.ReLU(inplace=True),

        #nn.ConvTranspose2d(ch*64, ch*16, 2, stride=1),
        #nn.ReLU(inplace=True),
        #torch.nn.GroupNorm(1,ch*16),
        #nn.BatchNorm2d(ch*16),

        nn.ConvTranspose2d(ch*16, ch*8, 5, stride=2),
        nn.ReLU(inplace=True),
        #torch.nn.GroupNorm(1,ch*8),
        #nn.BatchNorm2d(ch*8),


        nn.ConvTranspose2d(ch*8, ch*4, 5, stride=2),
        nn.ReLU(inplace=True),
        #torch.nn.GroupNorm(1,ch*4),
        #nn.BatchNorm2d(ch*4),

        nn.ConvTranspose2d(ch*4, ch*2, 5, stride=2),
        #GDN(ch*2, device),
        nn.ReLU(inplace=True),
        #torch.nn.GroupNorm(1,ch*2),
        #nn.BatchNorm2d(ch*2),

        nn.ConvTranspose2d(ch*2, ch, 6, stride=2),
        #GDN(ch, device),
        nn.ReLU(inplace=True),
        #torch.nn.GroupNorm(1,ch),
        #nn.BatchNorm2d(ch),

        nn.ConvTranspose2d(ch, 3, 6, stride=2),
)

# ReLU is sharper than GDN.
# Bigger kernels seem to help.
# At around SSIM=0.000026 minor stars start popping up, as well as Webb artifacts, grain still invisible
# Arch above has a latent space of 256x1x1 on 64 128x128 tiles, which means theoretically compressed size would be 64*256*32bits = 64kB
# Original image size in this case is 1024*1024*3 = ~3MB. Unfortunately the model itself is 10 MB :)
# At 0.000020 even the smallest stars become visible in the reconstruction. This seems like a good level for a usable compressor.
# This level was reached around epoch 6000 (and training stopped improving).
# [6419] l=0.0000, abs=0.0860, perc=0.00002301 (min=0.00001973)

torch.save(net[16:],'models/test')

net = net.to(device)
print(net(iset[0].unsqueeze(0)).shape)
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


#%%

opt = torch.optim.Adam(net.parameters(), lr=0.001)

perc = MS_SSIM(data_range=1, size_average=True, channel=3)
abs = nn.L1Loss()

perc_w = 1.0
abs_w = 1.0

net = net.to(device)
net.train()
early_stop = 0
report_steps = 1
image_steps = 20
for epoch in range(999999):
    running_loss = 0.0
    running_perc = 0.0
    running_abs = 0.0

    for i,t in enumerate(train,0):
        opt.zero_grad()

        o = net(t)

        perc_loss = 1.0 - ssim(o, t)
        abs_loss = abs(o, t)

        loss = perc_w*perc_loss + abs_w*abs_loss
        loss.backward()
        opt.step()

        running_loss += loss.item()
        running_abs += abs_loss.item()
        running_perc += perc_loss.item()

    if epoch%report_steps==(report_steps-1):
        print("[%d] l=%.4f, abs=%.4f, perc=%.8f (min=%.8f)" % (epoch, running_loss, running_abs, running_perc, min_loss))
    if epoch%image_steps==(image_steps-1):
        net.eval()
        show_tile(iset[1], net(iset[1].unsqueeze(0)).squeeze(0), size=5)
        net.train()

    if running_loss<min_loss:
        early_stop = 0
        min_loss = running_loss
        if running_loss<0.001:
            torch.save(net, 'models/01')
    else:
        early_stop += 1

    if early_stop>250:
        break

    running_loss = 0.0
    running_abs = 0.0
    running_perc = 0.0

#%%
net = torch.load('models/01')
net.eval()

show_img(iset, iset[:], size=15)
show_img(iset, net(iset[:]), size=15)


# Next:
# - deal with frames by tile overlap
# - try smaller tile (64x64)
# - increase amount of tiles
# - generally a better trade-off needs to be found because currently the model size destroys all savings.
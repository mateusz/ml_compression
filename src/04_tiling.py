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
name = '04b'

"""
Closest approach to a viable solution so far (still far from the real thing).
This also uses Tiler, which is my approach at implementing tiling reassembly
(which is non-ideal). Using something like
https://github.com/Mr-TalhaIlyas/EMPatches would be possibly better.
"""

#%%
class ImgSet(Dataset):
    def __init__(self, path, tile=256, pad=4, deadzone=2, htiles=-1, wtiles=-1):
        super().__init__()
        self.tile=tile
        self.pad=pad
        self.deadzone=deadzone

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img = transform(image)
        if htiles>0 and wtiles>0:
            img = img[:,:htiles*tile,:wtiles*tile]

        self.tiler = slib.Tiler(img, tile, pad, deadzone).to(device)
        self.img = self.tiler.unfold()

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        return self.img[idx]
    
    def save(self, img, path):
        folded = self.tiler.fold(img)
        cropped = self.tiler.crop_to_orig(folded)
        save_image(cropped, path)

    def show(self, img, size=5):
        folded = self.tiler.fold(img)
        cropped = self.tiler.crop_to_orig(folded)

        tile = cropped.permute(1,2,0).clamp(0.0,1.0).detach().cpu()

        fig,ax = plt.subplots(figsize=(size,size))
        ax.axis('off')
        ax.imshow(tile)
        plt.show()

tile=128
batch_size=16
pad=8
deadzone=4
htiles = wtiles = 32
iset = ImgSet('data/STScI-01GA76Q01D09HFEV174SVMQDMV.png', tile=tile, pad=pad, deadzone=deadzone, htiles=htiles, wtiles=wtiles)
print(iset.img.shape)

train = DataLoader(iset, batch_size=batch_size, shuffle=True)
print(len(train))

#%%

class Network(nn.Module):
    def __init__(self, N=128, pad=2):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.encode = nn.Sequential(
            nn.Conv2d(3, N, stride=2, kernel_size=pad*2+1, padding=pad),
            GDN(N),
            nn.Conv2d(N, N, stride=2, kernel_size=5, padding=2),
            GDN(N),
            nn.Conv2d(N, N, stride=2, kernel_size=5, padding=2),
        )

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(N, N, kernel_size=5, padding=2, output_padding=1, stride=2),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N, kernel_size=5, padding=2, output_padding=1, stride=2),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, 3, kernel_size=5, padding=2, output_padding=1, stride=2),
        )

    def forward(self, x):
       y = self.encode(x)
       y_hat, y_likelihoods = self.entropy_bottleneck(y)
       x_hat = self.decode(y_hat)
       return x_hat, y_likelihoods

net = Network(N=32, pad=pad).to(device)
print(net(iset[0].unsqueeze(0))[0].shape)
print(sum([l.numel() for l in net.parameters()]))
min_loss = 9999999

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

#%%

# 0.001 converges within tens of epochs but likes to explode sometimes
# 0.0001 converges around 200 epochs, and explodes less.
opt = torch.optim.Adam((net.encode+net.decode).parameters(), lr=0.0001)
# Example result for lr=0.001 and gradient clip=1, bpp_w=0.05
# [39] l=0.05493126, bpp=0.2067, perc=0.04459714, aux=49.8670 (min=0.05524260)
# Example result for lr=0.0001 and gradient clip=0.5, bpp_w=0.03
# [250] l=0.04610723, bpp=0.2569, perc=0.03840080, aux=98.3763 (min=0.04609825)
# Example result for lr=0.0001 and gradient clip=0.5, bpp_w=0.08
# [272] l=0.06151399, bpp=0.1345, perc=0.05075744, aux=99.1039 (min=0.06151295)

auxopt = torch.optim.Adam(net.entropy_bottleneck.parameters(), lr=0.01)

msssim = MS_SSIM(data_range=1, size_average=True, channel=3)

# Good loss here is around 0.02-0.03
ssim = SSIM(data_range=1, size_average=True, channel=3)

mse = torch.nn.MSELoss()

perc_w = 1.0
bpp_w = 0.08
# bpp_w=0.03 see 04a
# bpp_w=0.08 see 04b

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
            # Training tends to explode after a while
            clip_gradient(opt, 0.5)
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
                #compare_tiles(iset[1], net(iset[1].unsqueeze(0))[0].squeeze(0), size=5)
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

collect_out = slib.decompress_file(net, 'out/test-%s.parquet' % name, shape, batch_size).to(device)
iset.save(collect_out, 'out/test-%s.png' % name)

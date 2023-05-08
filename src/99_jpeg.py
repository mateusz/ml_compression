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
from torchvision.utils import save_image
import os
import io
import PIL
from enum import Enum
from pytorch_msssim import MS_SSIM as MS_SSIM
from skimage.metrics import peak_signal_noise_ratio as psnr
import src.lib as lib

PIL.Image.MAX_IMAGE_PIXELS = None

#%%

orig = PIL.Image.open('data/STScI-01GA76Q01D09HFEV174SVMQDMV.png')
orig.crop((512,0,1024,512)).save('out/orig-tile.png')
cropped = orig.crop((0,0,4096,4096))
cropped.save('out/test-orig.png')
cropped.save('out/test-jpeg95.jpg', quality=95)
cropped.save('out/test-jpeg70.jpg', quality=70)
cropped.save('out/test-jpeg50.jpg', quality=50)
cropped.save('out/test-jpeg20.jpg', quality=20)
orig = None
cropped = None

#%%
class Format(Enum):
    PIL = 1
    PARQUET = 2

class ImData():
    def __init__(self, d, format):
        if format==Format.PIL:
            data = PIL.Image.open(d)
            t = transforms.ToTensor()
            self.tensor = t(data)
            self.size = os.path.getsize(d)
        elif format==Format.PARQUET:
            data = PIL.Image.open('out/%s.png' % d)
            t = transforms.ToTensor()
            self.tensor = t(data)
            self.size = os.path.getsize('out/%s.parquet' % d)

        self.w, self.h = data.size

    def numpy(self):
        return self.tensor.permute(1,2,0).numpy()

def get_metrics(orig:ImData, comp:ImData):
    msssim = MS_SSIM(data_range=1, size_average=True, channel=3)

    m_psnr = psnr(orig.numpy(), comp.numpy())
    m_msssim = msssim(orig.tensor.unsqueeze(0), comp.tensor.unsqueeze(0)).numpy()
    rate = comp.size/orig.size
    bpp = (comp.size*8)/(comp.w*comp.h)

    return m_psnr, m_msssim, rate, bpp

orig = ImData('out/test-orig.png', Format.PIL)
#jpeg95 = ImData('out/test-jpeg95.jpg', Format.PIL)
#jpeg70 = ImData('out/test-jpeg70.jpg', Format.PIL)
#jpeg50 = ImData('out/test-jpeg50.jpg', Format.PIL)
#jpeg20 = ImData('out/test-jpeg20.jpg', Format.PIL)
#zero3c = ImData('test-03c', Format.PARQUET)
#zero3d = ImData('test-03d', Format.PARQUET)
#zero3e = ImData('test-03e', Format.PARQUET)
zero3f = ImData('test-03f', Format.PARQUET)
#zero4a = ImData('test-04a', Format.PARQUET)
zero4b = ImData('test-04b', Format.PARQUET)

#print("orig: psnr=%.2f, ms-ssim=%.6f, rate=%.4f, bpp=%.2f" % get_metrics(orig, orig))
#print("jpeg95: psnr=%.2f, ms-ssim=%.6f, rate=%.4f, bpp=%.2f" % get_metrics(orig, jpeg95))
#print("jpeg70: psnr=%.2f, ms-ssim=%.6f, rate=%.4f, bpp=%.2f" % get_metrics(orig, jpeg70))
#print("jpeg50: psnr=%.2f, ms-ssim=%.6f, rate=%.4f, bpp=%.2f" % get_metrics(orig, jpeg50))
#print("jpeg20: psnr=%.2f, ms-ssim=%.6f, rate=%.4f, bpp=%.2f" % get_metrics(orig, jpeg20))
#print("02a: psnr=%.2f, ms-ssim=%.6f, rate=%.4f, bpp=%.2f" % get_metrics(orig, zero2a))
#print("03c: psnr=%.2f, ms-ssim=%.6f, rate=%.4f, bpp=%.2f" % get_metrics(orig, zero3c))
#print("03d: psnr=%.2f, ms-ssim=%.6f, rate=%.4f, bpp=%.2f" % get_metrics(orig, zero3d))
#print("03e: psnr=%.2f, ms-ssim=%.6f, rate=%.4f, bpp=%.2f" % get_metrics(orig, zero3e))
print("03f: psnr=%.2f, ms-ssim=%.6f, rate=%.4f, bpp=%.2f" % get_metrics(orig, zero3f))
#print("04a: psnr=%.2f, ms-ssim=%.6f, rate=%.4f, bpp=%.2f" % get_metrics(orig, zero4a))
#print("04b: psnr=%.2f, ms-ssim=%.6f, rate=%.4f, bpp=%.2f" % get_metrics(orig, zero4b))

#orig: psnr=inf, ms-ssim=1.000000, rate=1.0000, bpp=9.53
#jpeg95: psnr=40.87, ms-ssim=0.990100, rate=0.1568, bpp=1.49
#03c: psnr=37.69, ms-ssim=0.979726, rate=0.0187, bpp=0.18
#jpeg70: psnr=37.24, ms-ssim=0.973139, rate=0.0416, bpp=0.40
#04a: psnr=36.58, ms-ssim=0.986462, rate=0.0316, bpp=0.30
#03f: psnr=36.47, ms-ssim=0.970446, rate=0.0148, bpp=0.14
#03e: psnr=36.41, ms-ssim=0.971706, rate=0.0146, bpp=0.14
#03d: psnr=33.96, ms-ssim=0.961352, rate=0.0133, bpp=0.13
#jpeg50: psnr=36.36, ms-ssim=0.963297, rate=0.0299, bpp=0.29
#04b: psnr=34.40, ms-ssim=0.973757, rate=0.0168, bpp=0.16
#jpeg20: psnr=34.05, ms-ssim=0.922156, rate=0.0189, bpp=0.18


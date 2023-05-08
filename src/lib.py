import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

# TODO store wtiles, htiles, tile and shape

def decompress_file(model, fname, shape, batch_size):
    if type(model)==str:
        net = torch.load(model)
        net.eval()
        net.entropy_bottleneck.update()
    else:
        net = model

    collect_out = torch.Tensor()
    f = pd.read_parquet(fname)
    for chunk in range(0, len(f), batch_size):
        chunkdata = f.iloc[chunk:chunk+batch_size]['chunk'].to_list()
        d = net.entropy_bottleneck.decompress(chunkdata, shape)
        o = net.decode(d).detach().cpu()
        collect_out = torch.concat([collect_out, o])
    
    return collect_out


class Tiler():
    """
    Tiler splits an tensor representing image into tiles.
    It can also rebuild image from tiles.
    Note the interpolation doesn't work well due to corner cases,
    so for now use only where deadzone==overlap/2.
    """

    def __init__(self, input, kern, overlap, deadzone):
        super().__init__()
        self.device = 'cpu'
        self.kern = kern
        self.overlap = overlap
        self.deadzone = deadzone
        self.orig_h_edge = input.size(1)
        self.orig_w_edge = input.size(2)
        self.h_edge = input.size(1)
        self.w_edge = input.size(2)
        self.ch = input.size(0)
        self.stride = kern-overlap
        self.data = input
        self.unfolded = False

        self.h_pad = self.get_expanded_edge(self.h_edge) - self.h_edge
        self.w_pad = self.get_expanded_edge(self.w_edge) - self.w_edge

        self.data = F.pad(self.data, (0,self.h_pad,0,self.w_pad), "constant", 0.0)
        self.h_edge=self.data.size(1)
        self.w_edge=self.data.size(2)

        self.w_tiles=(self.w_edge-self.overlap)//(self.stride)
        self.h_tiles=(self.h_edge-self.overlap)//(self.stride)

    def to(self, device):
        self.device = device
        self.data = self.data.to(device)
        return self

    def get_expanded_edge(self, edge):
        c = math.ceil((edge-self.overlap)/self.stride)
        newedge = c*self.stride+self.overlap
        return int(newedge)

    def unfold(self):
        unfolded = F.unfold(self.data, kernel_size=self.kern, stride=self.stride)
        unfolded = unfolded.permute(1,0).reshape(-1,self.ch,self.kern,self.kern)

        self.tile_count = unfolded.size(0)
        assert(self.tile_count==self.w_tiles*self.h_tiles)

        return unfolded

    def get_weights(self):
        # Calculate weights mask for interpolated merging.
        # Single tile mask with overlap 2 and kernel 5 will look like this (. is 0.33, : is 0.66 and 1 is 1)
        # .....
        # .:::.
        # .:1:.
        # .:::.
        # .....
        core_size = self.kern-2*self.overlap
        w=torch.ones(self.ch, core_size, core_size).to(self.device)
        # Step through interpolation
        for i in range(0, self.deadzone, 1):
            w = F.pad(w, (1,1,1,1), "constant", 1.0)

        for ring in torch.linspace(1.0, 0.0, self.overlap-(2*self.deadzone)+2)[1:-1]:
            w = F.pad(w, (1,1,1,1), "constant", ring)

        for i in range(0, self.deadzone, 1):
            w = F.pad(w, (1,1,1,1), "constant", 0.0)

        # Repeat tile mask over all batches.
        w = w.unsqueeze(0).repeat(self.tile_count,1,1,1)

        return w

    def fold(self, data):
        w = self.get_weights()
        
        folded = data*w
        folded = folded.reshape(self.tile_count, -1).permute(1,0)
        folded = F.fold(folded, output_size=(self.h_edge,self.w_edge), kernel_size=self.kern, stride=self.stride)

        return folded

    def crop_to_orig(self, data):
        return data[:,:self.orig_h_edge,:self.orig_w_edge]
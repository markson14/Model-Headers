import torch
from torch import nn
import torch.nn.functional as F
import numpy
import math


def _make_layer(inp_dim, out_dim, modules):
    layers = [residual(inp_dim, out_dim)]
    layers += [residual(out_dim, out_dim) for _ in range(1, modules)]
    return nn.Sequential(*layers)


def _make_layer_revr(inp_dim, out_dim, modules):
    layers = [residual(inp_dim, inp_dim) for _ in range(modules - 1)]
    layers += [residual(inp_dim, out_dim)]
    return nn.Sequential(*layers)


class saccade_module(nn.Module):
    def __init__(self, n, dims, modules, make_up_layer=_make_layer, make_hg_layer=_make_layer,
                 make_low_layer=_make_layer, make_hg_layer_revr=_make_layer_revr,  make_merge_layer=_make_merge_layer):
        super(saccade_module, self).__init__()

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.n = n
        self.up1 = make_up_layer(curr_dim, curr_dim, curr_mod)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.low1 = make_hg_layer(curr_dim, next_dim, curr_mod)
        self.low2 = saccade_module(
            n - 1, dims[1:], modules[1:],
            make_up_layer=make_up_layer,
            make_pool_layer=nn.MaxPool2d,
            make_hg_layer=make_hg_layer,
            make_low_layer=make_low_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_unpool_layer=F.interpolate,
            make_merge_layer=make_merge_layer
        ) if n > 1 else make_low_layer(next_dim, next_dim, next_mod)
        self.low3 = make_hg_layer_revr(next_dim, curr_dim, curr_mod)
        self.up2 = F.interpolate
        self.merg = make_merge_layer(curr_dim)

    def forward(self, x):
        up1 = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        if self.n > 1:
            low2, mergs = self.low2(low1)
        else:
            low2, mergs = self.low2(low1), []
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        merg = self.merg(up1, up2)
        mergs.append(merg)
        return merg, mergs

import torch
import torch.nn as nn
from torch.legacy import nn as old_nn
from torch.legacy.nn.Sequential import Sequential as old_Sequential
from torch.utils.serialization import load_lua
from glcic import glcic

A = load_lua('completionnet_places2.t7', long_size=8).model
B = glcic(in_ch=4, out_ch=3, ch=64)

A_layers = list(m for m in A.modules if isinstance(m, (old_nn.SpatialConvolution, old_nn.SpatialFullConvolution, old_nn.SpatialDilatedConvolution, old_nn.SpatialBatchNormalization)))
B_layers = list(m for m in B.modules() if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)))

for m1, m2 in zip(A_layers, B_layers):
    # m1, m2 = A_layers[0], B_layers[0]
    if isinstance(m2, (nn.Conv2d, nn.ConvTranspose2d)):
        m2.weight.data.copy_(m1.weight)
        m2.bias.data.copy_(m1.bias)
    elif isinstance(m2, nn.BatchNorm2d):
        m2.running_var.copy_(m1.running_var)
        m2.running_mean.copy_(m1.running_mean)
        m2.weight.data.copy_(m1.weight)
        m2.bias.data.copy_(m1.bias)

torch.save({'model': B.state_dict()}, 'completionnet_places2.pt')

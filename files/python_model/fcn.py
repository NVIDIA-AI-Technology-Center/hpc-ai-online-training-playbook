# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import torch
from torch import nn
import torch.nn.functional as F

def weight_init(model):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

class FCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 skip = False,
                 input_pad = 16,
                 output_crop = 2,
                 filter_sizes = [64, 128, 256, 256, 128],
                 kernel_sizes = [5, 3, 3, 3, 3, 3],
                 negative_slope = 0.01):
        super(FCN, self).__init__()

        # padding function
        self.crop = output_crop // 2
        self.pad = (input_pad//2, input_pad//2, input_pad//2, input_pad//2)
        
        # first layer
        layers = [nn.Conv2d(in_channels = in_channels,
                            out_channels = filter_sizes[0],
                            kernel_size = kernel_sizes[0],
                            padding = 0,
                            bias = False),
                  nn.BatchNorm2d(filter_sizes[0]),
                  nn.LeakyReLU(negative_slope=negative_slope)]

        filter_sizes.append(out_channels)
        for idl in range(1, len(filter_sizes)):
            layers.append(nn.Conv2d(in_channels = filter_sizes[idl-1],
                                    out_channels = filter_sizes[idl],
                                    kernel_size = kernel_sizes[idl],
                                    padding = 0,
                                    bias = False if (idl < len(filter_sizes)-1) else True))
            if idl < len(filter_sizes)-1:
                layers.append(nn.BatchNorm2d(filter_sizes[idl]))
                layers.append(nn.LeakyReLU(negative_slope=negative_slope))

        self.fwd = nn.Sequential(*layers)
        self.skip = skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xp = F.pad(x, self.pad, mode='circular')
        xc = self.fwd(xp)
        B, C, H, W = xc.shape
        out = xc[..., self.crop:H-self.crop, self.crop:W-self.crop]

        if self.skip:
            out = out + x
        
        return out

# script model:
#device = torch.device("cuda:0")
device = torch.device("cpu")
model = FCN(3, 3, skip=False).to(device)

# we need to initialize the model:
weight_init(model)

jmodel = torch.jit.script(model)

inp = torch.ones((1, 3, 125, 141), dtype=torch.float32, device=device)
out = jmodel(inp)

# printing
print(model)
print(out.shape)

# save model
torch.jit.save(jmodel, "./files/python_model/cans_fcn.pt") 


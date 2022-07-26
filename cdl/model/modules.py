import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import AttrDict


class Residual(nn.Module):
    # convolution residual block
    def __init__(self, channel, kernel):
        super().__init__()
        self.conv = nn.Conv2d(channel, channel, kernel, padding=kernel // 2)
        self.scalar = nn.Parameter(torch.tensor(0, dtype=torch.float32))

    def forward(self, x):
        out = self.conv(F.leaky_relu(x, negative_slope=0.02))
        return x + out * self.scalar


class Reshape(nn.Module):
    # reshape last dim to (c, h, w)
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.dim = np.prod(self.shape)

    def forward(self, x):
        assert x.shape[-1] == self.dim
        return x.view(*x.shape[:-1], *self.shape)


def get_backbone(params, encoding, verbose=False):
    encoder_params = params.encoder_params
    decoder_params = params.decoder_params

    image_spec = np.concatenate([obs for obs in params.obs_spec.values() if obs.ndim == 3], axis=0).shape

    c = h = w = dim = None
    if encoding:
        c, h, w = image_spec
        assert h == w
    else:
        dim = params.encoder_params.feature_dim

    def get_shape():
        if encoding:
            if dim is None:
                shape = (c, h, w)
            else:
                shape = dim
        else:
            if c is None:
                shape = dim
            else:
                shape = (c, h, w)
        return shape

    module_list = []
    modules = encoder_params.modules if encoding else decoder_params.modules
    for i, mod_params in enumerate(modules):
        mod_params = AttrDict(mod_params)
        module_type = mod_params.type
        mod_params.pop("type")

        if verbose:
            if i == 0:
                print("encoder" if encoding else "decoder")
            print("{}-th module:".format(i + 1), module_type, mod_params)
            print("input shape:", get_shape())

        if module_type == "conv":
            if mod_params.channel is None:
                assert not encoding and i == len(modules) - 1
                mod_params.channel = image_spec[0]
            module = nn.Conv2d(c, mod_params.channel, mod_params.kernel, mod_params.stride, mod_params.kernel // 2)
            w = w // mod_params.stride
            h = h // mod_params.stride
            c = mod_params.channel
        elif module_type == "residual":
            module = Residual(c, mod_params.kernel)
        elif module_type == "avg_pool":
            module = nn.AvgPool2d(kernel_size=mod_params.kernel)
            w = w // mod_params.kernel
            h = h // mod_params.kernel
        elif module_type == "upsample":
            module = nn.Upsample(scale_factor=mod_params.scale_factor, mode="bilinear", align_corners=False)
            w = w * mod_params.scale_factor
            h = h * mod_params.scale_factor
        elif module_type == "flatten":
            assert dim is None
            module = nn.Flatten(start_dim=-3, end_dim=-1)
            dim = w * h * c
        elif module_type == "reshape":
            assert c is None and h is None and w is None
            assert dim == np.prod(mod_params.shape)
            module = Reshape(mod_params.shape)
            c, h, w = mod_params.shape
        elif module_type == "linear":
            module = nn.Linear(dim, mod_params.dim)
            dim = mod_params.dim
        elif module_type == "layer_norm":
            module = nn.LayerNorm(dim)
        elif module_type == "tanh":
            module = nn.Tanh()
        elif module_type == "relu":
            module = nn.ReLU()
        elif module_type == "leaky_relu":
            module = nn.LeakyReLU(negative_slope=mod_params.alpha)
        else:
            raise NotImplementedError

        module_list.append(module)

        if verbose:
            print("output shape:", get_shape())
            if i == len(modules) - 1:
                print()

    if encoding:
        output_shape = dim
    else:
        output_shape = (c, h, w)

    return nn.Sequential(*module_list), output_shape

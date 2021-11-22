"""
"""
import argparse
from neuralcompress.models.blocks import (
    encoder_residual_block,
    decoder_residual_block
)
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(
        self,
        num_layers,
        activ=None,
        norm=None
    ):
        super().__init__()
        conv_args = {
            'in_channels': 8,
            'out_channels': 8,
            'kernel_size': 4,
            'stride': 2,
            'padding': 1
        }
        layers = []
        for _ in range(num_layers):
            layer = encoder_residual_block(conv_args, activ, norm)
            layers.append(layer)
        self.layers = nn.Sequential(*layers)

    def forward(self, x_input):
        return self.layers(x_input)


encoder = Encoder(3, 'relu', 'batch')
tensor_in = torch.randn(32, 8, 16, 16, 16)
tensor_out = encoder(tensor_in)
print(tensor_out.shape)

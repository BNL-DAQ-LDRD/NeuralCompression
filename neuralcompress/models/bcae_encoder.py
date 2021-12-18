"""
User need to define encoder here
"""

import torch.nn as nn
from neuralcompress.models.bcae_blocks import (
    single_block,
    encoder_residual_block,
)

class Encoder(nn.Module):
    """
    Encoder with a few downsampling layers plus an output layer.
    """
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        input_channels,
        conv_args_list,
        activ,
        norm_fn,
        output_channels,
        rezero=True
    ):
        """
        Input:
            - input_channels (int): input_channels of the first convolution
                layers
            - conv_args_list (list of dictionary): arguments for the
                convolution/downsampling layers. Each entry in the list is
                a dictionary contains the following keys:
                - out_channels;
                - kernel_size;
                - stride;
                - padding;
            - activ: activation layer;
            - norm: normalization function (a normalization function without
                parameter. Need to be initialized with parameter.)
            - output_channels (int): out_channels in the output layer.
        """
        super().__init__()

        # Downsampling layers
        self.layers, in_ch = nn.Sequential(), input_channels
        for idx, conv_args in enumerate(conv_args_list):
            conv_args['in_channels'] = in_ch

            layer = encoder_residual_block(
                conv_args,
                activ,
                norm_fn(conv_args['out_channels']),
                rezero=rezero
            )

            self.layers.add_module(f'encoder_block_{idx}', layer)
            in_ch = conv_args['out_channels']

        # Encoder output layer
        block_args = {
            'in_channels'  : in_ch,
            'out_channels' : output_channels,
            'kernel_size'  : 3,
            'padding'      : 1
        }
        output_layer = single_block('conv', block_args)
        self.layers.add_module('encoder_output', output_layer)

    def forward(self, input_x):
        """
        input_x shape: (N, C, D, H, W)
            - N = batch_size;
            - C = input_channels;
            - D, H, W: the three spatial dimensions
        """
        return self.layers(input_x)


def get_bcae_encoder():
    """
    User need to provide such a function.
    They should adjust the parameter here.
    """
    conv_1 = {
        'out_channels': 8,
        'kernel_size' : [4, 3, 3],
        'padding'     : [1, 0, 1],
        'stride'      : [2, 2, 1]
    }
    conv_2 = {
        'out_channels': 16,
        'kernel_size' : [4, 4, 3],
        'padding'     : [1, 1, 1],
        'stride'      : [2, 2, 1]
    }
    conv_3 = {
        'out_channels': 32,
        'kernel_size' : [4, 4, 3],
        'padding'     : [1, 1, 1],
        'stride'      : [2, 2, 1]
    }
    conv_4 = {
        'out_channels': 32,
        'kernel_size' : [4, 3, 3],
        'padding'     : [1, 0, 1],
        'stride'      : [2, 2, 1]
    }

    image_channels  = 1
    conv_args_list  = [conv_1, conv_2, conv_3, conv_4]
    activ           = nn.LeakyReLU(negative_slope=.2)
    norm_fn         = nn.InstanceNorm3d
    code_channels   = 8
    rezero          = True

    return Encoder(
        image_channels,
        conv_args_list,
        activ,
        norm_fn,
        code_channels,
        rezero
    )

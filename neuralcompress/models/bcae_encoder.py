"""
User need to define encoder here
"""
import torch.nn as nn
from neuralcompress.models.bcae_blocks import (
    single_block,
    encoder_residual_block,
)

class BCAEEncoder(nn.Module):
    """
    Encoder with a few downsampling layers plus an output layer.
    """

    # class constants for default settings:
    CONV_1 = {
        'out_channels': 8,
        'kernel_size' : [4, 3, 3],
        'padding'     : [1, 0, 1],
        'stride'      : [2, 2, 1]
    }
    CONV_2 = {
        'out_channels': 16,
        'kernel_size' : [4, 4, 3],
        'padding'     : [1, 1, 1],
        'stride'      : [2, 2, 1]
    }
    CONV_3 = {
        'out_channels': 32,
        'kernel_size' : [4, 4, 3],
        'padding'     : [1, 1, 1],
        'stride'      : [2, 2, 1]
    }
    CONV_4 = {
        'out_channels': 32,
        'kernel_size' : [4, 3, 3],
        'padding'     : [1, 0, 1],
        'stride'      : [2, 2, 1]
    }

    IMAGE_CHANNELS  = 1
    CONV_ARGS_LIST  = (CONV_1, CONV_2, CONV_3, CONV_4)
    ACTIV           = nn.LeakyReLU(negative_slope=.2)
    NORM_FN         = nn.InstanceNorm3d
    CODE_CHANNELS   = 8
    REZERO          = True


    # pylint: disable=too-many-arguments
    def __init__(
        self,
        input_channels  = IMAGE_CHANNELS,
        conv_args_list  = CONV_ARGS_LIST,
        activ           = ACTIV,
        norm_fn         = NORM_FN,
        output_channels = CODE_CHANNELS,
        rezero          = REZERO
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
        norm = norm_fn(output_channels)
        output_layer = single_block('conv', block_args, activ, norm)
        self.layers.add_module('encoder_output', output_layer)

    def forward(self, input_x):
        """
        input_x shape: (N, C, D, H, W)
            - N = batch_size;
            - C = input_channels;
            - D, H, W: the three spatial dimensions
        """
        return self.layers(input_x)

if __name__ == "__main__":
    print("This is the main of bcae_encoder.py")
    encoder = BCAEEncoder()
    print(encoder)

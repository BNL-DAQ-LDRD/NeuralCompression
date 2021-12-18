"""
User need to define decoder here
"""

import torch.nn as nn
from neuralcompress.models.bcae_blocks import (
    single_block,
    decoder_residual_block,
)

class DecoderOneHead(nn.Module):
    """
    Decoder with a few upsampling layers plus an output layer.
    """
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        input_channels,
        deconv_args_list,
        activ,
        norm_fn,
        output_channels,
        output_activ,
        output_norm,
        rezero=True
    ):
        """
        Input:
            - input_channels (int): input_channels of the first deconvolution
                layers.
            - conv_args_list (list of dictionary): arguments for the
                deconvolution/upsampling layers. Each entry in the list is
                a dictionary contains the following keys:
                - out_channels;
                - kernel_size;
                - stride;
                - padding;
                - output_padding;
            - activ: activation layer;
            - norm: normalization function (a normalization function without
                parameter. Need to be initialized with parameter.)
            - output_channels (int): out_channels in the output layer.
            - output_activ: output activation layer;
            - output_norm: normalization function for the output layer.
        """
        super().__init__()

        # Upsampling layers
        self.layers, in_ch = nn.Sequential(), input_channels
        for idx, deconv_args in enumerate(deconv_args_list):
            deconv_args['in_channels'] = in_ch

            layer = decoder_residual_block(
                deconv_args,
                activ,
                norm_fn(deconv_args['out_channels']),
                rezero=rezero
            )

            self.layers.add_module(f'decoder_block_{idx}', layer)
            in_ch = deconv_args['out_channels']

        # Decoder output layer
        block_args = {
            'in_channels'  : in_ch,
            'out_channels' : output_channels,
            'kernel_size'  : 3,
            'padding'      : 1
        }
        output_layer = single_block(
            block_type = 'conv',
            block_args = block_args,
            activ      = output_activ,
            norm       = output_norm
        )
        self.layers.add_module('decoder_output', output_layer)

    def forward(self, input_x):
        """
        input_x shape: (N, C, D, H, W)
            - N = batch_size;
            - C = input_channels;
            - D, H, W: the three spatial dimensions
        """
        return self.layers(input_x)


class Decoder(nn.Module):
    """
    BCAE decoder with two heads.
    """
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        input_channels,
        deconv_args_list,
        activ,
        norm_fn,
        output_channels,
        rezero=True
    ):
        """
        Input:
            - code_channels (int): number of channels of the code.
            - deconv_args_list (list of dictionary): arguments for the
                deconvolution/upsampling layers. Each entry in the list is
                a dictionary contains the following keys:
                - out_channels;
                - kernel_size;
                - stride;
                - padding;
                - output_padding;
            - activ: activation layer;
            - norm: normalization function (a normalization function without
                parameter. Need to be initialized with parameter.)
            - image_channels (int): number of channels of the input image.
        """
        super().__init__()

        args = {
            'input_channels'   : input_channels,
            'deconv_args_list' : deconv_args_list,
            'activ'            : activ,
            'norm_fn'          : norm_fn,
            'output_channels'  : output_channels
        }
        self.decoder_c = DecoderOneHead(
            **args,
            output_activ = nn.Sigmoid(),
            output_norm  = nn.Identity(),
            rezero       = rezero
        )
        self.decoder_r = DecoderOneHead(
            **args,
            output_activ = nn.Identity(),
            output_norm  = nn.Identity(),
            rezero       = rezero
        )

    def forward(self, code):
        """
        input_x shape: (N, C, D, H, W)
            - N = batch_size;
            - C = image_channels;
            - D, H, W: the three spatial dimensions
        """
        return self.decoder_c(code), self.decoder_r(code)


def get_bcae_decoder():
    """
    User need to provide such a function.
    They should adjust the parameter here.
    """

    deconv_1 = {
        'out_channels': 16,
        'kernel_size' : [4, 3, 3],
        'padding'     : [1, 0, 1],
        'stride'      : [2, 2, 1],
        'output_padding': 0
    }
    deconv_2 = {
        'out_channels': 8,
        'kernel_size' : [4, 4, 3],
        'padding'     : [1, 1, 1],
        'stride'      : [2, 2, 1],
        'output_padding': 0
    }
    deconv_3 = {
        'out_channels': 4,
        'kernel_size' : [4, 4, 3],
        'padding'     : [1, 1, 1],
        'stride'      : [2, 2, 1],
        'output_padding': 0
    }
    deconv_4 = {
        'out_channels': 2,
        'kernel_size' : [4, 3, 3],
        'padding'     : [1, 0, 1],
        'stride'      : [2, 2, 1],
        'output_padding': 0
    }

    image_channels   = 1
    deconv_args_list = [deconv_1, deconv_2, deconv_3, deconv_4]
    activ            = nn.LeakyReLU(negative_slope=.2)
    norm_fn          = nn.InstanceNorm3d
    code_channels    = 8
    rezero           = True

    return Decoder(
        code_channels,
        deconv_args_list,
        activ,
        norm_fn,
        image_channels,
        rezero
    )

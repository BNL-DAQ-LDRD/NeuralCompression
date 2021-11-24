"""
yhuang2@bnl.gov
"""

import torch
import torch.nn as nn
from neuralcompress.models.blocks import (
    single_block,
    encoder_residual_block,
    decoder_residual_block
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
        norm,
        output_channels,
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
            - activ (str or dictionary): type or parameters of the activation
                in the upsampling layers.
            - norm (str): normalization type.
            - output_channels (int): out_channels in the output layer.
        """
        super().__init__()

        # Downsampling layers
        self.layers, in_ch = nn.Sequential(), input_channels
        for idx, conv_args in enumerate(conv_args_list):
            conv_args['in_channels'] = in_ch
            layer = encoder_residual_block(conv_args, activ, norm)
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
        forward
        """
        return self.layers(input_x)


class Decoder(nn.Module):
    """
    Decoder with a few upsampling layers plus an output layer.
    """
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        input_channels,
        deconv_args_list,
        activ,
        norm,
        output_channels,
        output_activ = None,
        output_norm  = None,
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
            - activ (str or dictionary): type or parameters of the activation
                in the upsampling layers.
            - norm (str): normalization type.
            - output_channels (int): out_channels in the output layer.
            - output_activ (str or dictionary): type or parameters of the
                activation in the output layer.
            - output_norm (str): type of output layer.
        """
        super().__init__()

        self.half_mode = False

        # Upsampling layers
        self.layers, in_ch = nn.Sequential(), input_channels
        for idx, deconv_args in enumerate(deconv_args_list):
            deconv_args['in_channels'] = in_ch
            layer = decoder_residual_block(deconv_args, activ, norm)
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

    def set_half_mode(self, mode):
        """
        when half_mode is set to True, use half float
        """
        self.half_mode = mode

    def forward(self, input_x):
        """
        forward
        """
        if self.half_mode:
            input_x = input_x.type(torch.float16)
            with torch.cuda.amp.autocast():
                return self.layers(input_x)
        return self.layers(input_x)


# The Auto-Encoder
class BCAE(nn.Module):
    """
    An auto-encoder with one encoder and two decoders --
    one for segmentation and one for regression.
    """
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        image_channels,
        code_channels,
        conv_args_list,
        deconv_args_list,
        activ,
        norm,
        transform=True
    ):
        """
        Input:
            - image_channels (int): number of channels of the input image.
            - code_channels (int): number of channels of the code.
            - conv_args_list (list of dictionary): arguments for the
                convolution/downsampling layers. Each entry in the list is
                a dictionary contains the following keys:
                - out_channels;
                - kernel_size;
                - stride;
                - padding;
            - deconv_args_list (list of dictionary): arguments for the
                deconvolution/upsampling layers. Each entry in the list is
                a dictionary contains the following keys:
                - out_channels;
                - kernel_size;
                - stride;
                - padding;
                - output_padding;
            - activ (str or dictionary): type or parameters for the activation
                in encoder/decoder.
            - norm (str): type for the normalization in encoder/decoder .
            - transform (bool): whether to do input transform.
        """
        super().__init__()

        # Encoder
        self.encoder = Encoder(
            image_channels,
            conv_args_list,
            activ,
            norm,
            code_channels
        )

        # Decoders
        args = [
            code_channels,
            deconv_args_list,
            activ,
            norm,
            image_channels
        ]
        activ_clf = 'sigmoid'
        activ_reg = None if transform else nn.ReLU()
        self.decoder_c = Decoder(*args, output_activ=activ_clf)
        self.decoder_r = Decoder(*args, output_activ=activ_reg)
        self.decoder_c.set_half_model(False)
        self.decoder_r.set_half_model(False)

    def set_half_mode(self, mode):
        """
        when half_mode is set to True, use half float
        """
        self.decoder_c.set_half_mode(mode)
        self.decoder_r.set_half_mode(mode)

    def forward(self, input_x):
        """
        Forward
        """
        code = self.encoder(input_x)
        output_clf = self.decoder_c(code)
        output_reg = self.decoder_r(code)
        return output_clf, output_reg

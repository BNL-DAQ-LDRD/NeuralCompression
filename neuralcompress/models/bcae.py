"""
yhuang2@bnl.gov
"""

import torch
import torch.nn as nn
from blocks import (
    encoder_block,
    EncoderResidualBlock,
    decoder_block,
    DecoderResidualBlock
)

class DictList:
    def __init__(self, dictionary):
        list_length = None
        for val in dictionary.values():
            if list_length is None:
                list_length = len(val)
            assert list_length == len(val), \
                'lists in the dictionary must all have the same length'
        self.list_length = list_length
        self.dictionary = dictionary

    def __getitem__(self, i):
        return {key: val[i] for key, val in self.dictionary.items()}

    def __len__(self):
        return self.list_length


# Encoder
class CNNEncoder(nn.Module):
    """
    Encoder.
    """
    def __init__(
        self,
        input_channels,
        conv_args_list,
        activ,
        norm,
        output_layer,
    ):
        """
        Input:
        Output:
        """
        super().__init__()

        # Downsampling layers
        layers, in_ch = [], input_channels
        for conv_args in conv_args_list:
            conv_args['input_channels'] = in_ch
            layer = EncoderResidualBlock(conv_args, activ, norm)
            layers.append(layer)
            in_ch = conv_args['output_channels']

        layers.append(output_layer)
        self.layers = nn.Sequential(layers)

    def forward(self, input_x):
        return self.layers(input_x)


# Decoder
class CNNDecoder(nn.Module):
    """
    Decoder.
    """
    def __init__(
        self,
        input_channels,
        deconv_args_list,
        activ,
        norm,
        output_layer,
    ):
        """
        Input:
        Output:
        """
        super().__init__()

        self.half_mode = False

        layers, in_ch = [], input_channels
        for deconv_args in deconv_args_list:
            deconv_args['input_channels'] = in_ch
            layer = DecoderResidualBlock(deconv_args, activ, norm)
            layers.append(layer)
            in_ch = deconv_args['output_channels']

        layers.append(output_layer)
        self.layers = nn.Sequential(layers)

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
class CNNAE(nn.Module):
    """
    A double-headed auto-encoder
    """
    def __init__(
        self,
        image_channels,
        conv_args_list,
        encoder_output_channels,
        deconv_args_list,
        activ,
        norm,
        transform=True
):
        super().__init__()
        self.half_mode = False

        # Encoder output layer
        output_conv_args = {
            'input_channels'  : conv_args_list[-1]['output_channels'],
            'output_channels' : encoder_output_channels,
            'kernel_size'     : 1,
            'stride'          : 1,
            'padding'         : 0,
        }
        # Encoder
        self.encoder = CNNEncoder(
            image_channels,
            conv_args_list,
            activ,
            norm,
            encoder_block(output_conv_args, activ, norm)
        )

        # Decoder output layer
        output_deconv_args = {
            'input_channels'  : deconv_args_list[-1]['output_channels'],
            'output_channels' : image_channels,
            'kernel_size'     : 1,
            'stride'          : 1,
            'padding'         : 0,
            'output_padding'  : 0
        }
        activ_clf = 'sigmoid'
        activ_reg = None if transform else nn.ReLU()
        output_layer_clf = decoder_block(output_deconv_args, activ_clf)
        output_layer_reg = decoder_block(output_deconv_args, activ_reg)
        # Decoders
        args = [
            encoder_output_channels,
            deconv_args_list,
            activ,
            norm
        ]
        self.decoder_c = CNNDecoder(*args, output_layer_clf)
        self.decoder_r = CNNDecoder(*args, output_layer_reg)

    def set_half_mode(self, mode):
        """
        when half_mode is set to True, use half float
        """
        self.half_mode = mode

    def forward(self, input_x):
        """
        Forward
        """
        code = self.encoder(input_x)
        self.decoder_c.set_half_mode(self.hald_mode)
        self.decoder_r.set_half_mode(self.hald_mode)
        output_clf = self.decoder_c(code)
        output_reg = self.decoder_r(code)
        return output_clf, output_reg

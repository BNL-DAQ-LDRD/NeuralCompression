"""
Author:
    Yi Huang, yhuang2@bnl.gov
"""
import torch.nn as nn
from neuralcompress.torch.select import(
    get_norm_layer_fn,
    get_activ_layer,
)


def single_block(block_type, block_args, activ, norm):
    assert block_type in ['conv', 'deconv']

    if block_type == 'conv':
        layer = nn.Conv3d(**block_args)
    elif block_type == 'deconv':
        layer = nn.ConvTranspose3d(**block_args)

    activ   = get_activ_layer(activ)
    norm_fn = get_norm_layer_fn(norm)
    norm    = norm_fn(block_args['output_channels'])

    return nn.Sequential(layer, activ, norm)


def double_block(block_type, block_args, activ, norm):
    assert block_type in ['conv', 'deconv']

    if block_type == 'conv':
        layer_1 = nn.Conv3d(**block_args)
        layer_2 = nn.Conv3d(
            conv_args['output_channels'],
            conv_args['output_channels'],
            kernel_size = 3,
            stride      = 1,
            padding     = 1
        )
    elif block_type == 'deconv':
        layer_1 = nn.ConvTranspose3d(**block_args)
        layer_2 =




def double_block(block_type, block_args, avtiv, norm):


class ResidualBlock(nn.Module):
    def __init__(
        self,
        main_block,
        side_block,
        out_channels,
        activ=None,
        norm=None
    ):
        self.main_block = main_block
        self.side_block = side_block

        self.activ    = get_activ_layer(activ)
        norm_layer_fn = get_norm_layer_fn(norm)
        self.norm     = norm_layer_fn(out_channels)

    def forward(self, x_input):
        """
        forward
        """
        x_side = self.side_block(x_input)
        x_main = self.main_block(x_input)
        x_output = x_main + x_side
        return self.norm(self.activ(x_output))


# Encoder blocks
def encoder_block(conv_args, activ=None, norm=None):
    """
    Encoder block with convolution, normalization and activation.
    """
    conv_layer = nn.Conv3d(
        conv_args['input_channels'],
        conv_args['output_channels'],
        kernel_size = conv_args['kernel_size'],
        stride      = conv_args['stride'],
        padding     = conv_args['padding']
    )

    activ_layer   = get_activ_layer(activ)
    norm_layer_fn = get_norm_layer_fn(norm)
    norm_layer    = norm_layer_fn(conv_args['output_channels'])

    return nn.Sequential(conv_layer, activ_layer, norm_layer)


def encoder_block_double(conv_args, activ=None, norm=None):
    """
    Encoder block with convolution, normalization, activation,
    and an additional convolution.
    """
    conv_block = encoder_block(conv_args, activ=activ, norm=norm)
    conv_layer = nn.Conv3d(
        conv_args['output_channels'],
        conv_args['output_channels'],
        kernel_size = 3,
        stride      = 1,
        padding     = 1
    )

    return nn.Sequential(conv_block, conv_layer)


def encoder_residual_block(
    conv_args,
    activ=None,
    norm=None
):
    """
    Get an encoder residual block.
    """
    return ResidualBlock(
        main_block = encoder_block_double(conv_args, activ, norm)
        side_block = encoder_block(conv_args, activ, norm),
        activ      = activ
        norm       = norm
    )


# Decoder blocks
def decoder_block(deconv_args, activ=None, norm=None):
    """
    Decoder block with deconvolution, normalization, and activation.
    """
    deconv = nn.ConvTranspose3d(
        deconv_args['input_channels'],
        deconv_args['output_channels'],
        kernel_size     = deconv_args['kernel_size'],
        stride          = deconv_args['stride'],
        padding         = deconv_args['padding'],
        output_padding  = deconv_args['output_padding']
    )

    activ_layer   = get_activ_layer(activ)
    norm_layer_fn = get_norm_layer_fn(norm)
    norm_layer    = norm_layer_fn(deconv_args['output_channels'])

    return nn.Sequential(deconv, activ_layer, norm_layer)


def decoder_block_double(deconv_args, activ=None, norm=None):
    """
    Decoder block with deconvolution, normalization, activation,
    and an additional deconvolution layer.
    """
    deconv_block = decoder_block(deconv_args, activ=activ, norm=norm)
    deconv_layer = nn.ConvTranspose3d(
        deconv_args['output_channels'],
        deconv_args['output_channels'],
        kernel_size     = 3,
        stride          = 1,
        padding         = 1,
        output_padding  = deconv_args['output_padding']
    )

    return nn.Sequential(deconv_block, deconv_layer)



class DecoderResidualBlock(nn.Module):
    """
    Decoder residual block.
    """
    def __init__(self, deconv_args, activ=None, norm=None):
        super().__init__()

        self.side_block = decoder_block(deconv_args, activ, norm)
        self.main_block = decoder_block_double(deconv_args, activ, norm)

        self.activ    = get_activ_layer(activ)
        norm_layer_fn = get_norm_layer_fn(norm)
        self.norm     = norm_layer_fn(deconv_args['output_channels'])

    def forward(self, x_input):
        """
        forward
        """
        x_side = self.side_block(x_input)
        x_main = self.main_block(x_input)
        x_output = x_main + x_side
        return self.norm(self.activ(x_output))

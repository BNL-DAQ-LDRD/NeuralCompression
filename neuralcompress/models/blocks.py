"""
Author:
    Yi Huang, yhuang2@bnl.gov
"""
import torch
import torch.nn as nn
from neuralcompress.torch.select import(
    get_norm_layer_fn,
    get_activ_layer,
)


def single_block(block_type, block_args, activ=None, norm=None):
    """
    A convolution/deconvolution block
    with activation and normalization
    Input:
        - block_type (str): 'conv' for convolution and
            'deconv' for deconvolution;
        - block_args (dictionary): arguments for the
            convolution/deconvolution block;
        - activ (str or dictionary): config of activation
        - norm (str): normalization type;
    Output:
        An nn.Sequential object with convolution/deconvolution
        followed by activation and then by normalization.
    """
    assert block_type in ['conv', 'deconv']

    if block_type == 'conv':
        layer = nn.Conv3d(**block_args)
    elif block_type == 'deconv':
        layer = nn.ConvTranspose3d(**block_args)

    activ   = get_activ_layer(activ)
    norm_fn = get_norm_layer_fn(norm)
    norm    = norm_fn(block_args['out_channels'])

    return nn.Sequential(layer, activ, norm)


def double_block(block_type, block_args, activ=None, norm=None):
    """
    A double convolution/deconvolution block that contains
        - a convolution/deconvolution layer that does,
            if need, down or up sampling;
        - activation;
        - normalization;
        - an additional shape-invariant convolution/deconvolution layer.
    Input:
        - block_type (str): 'conv' for convolution and
            'deconv' for deconvolution;
        - block_args (dictionary): arguments for the
            convolution/deconvolution block;
        - activ (str or dictionary): config of activation
        - norm (str): normalization type;
    Output:
        An nn.Sequential object with two convolution/deconvolution layers,
        and activation and normalization in middle.
    """

    block_1 = single_block(block_type, block_args, activ, norm)
    # if block_type == 'conv':
    #     layer_fn = nn.Conv3d
    # else:
    #     layer_fn = nn.ConvTranspose3d
    layer_fn = nn.Conv3d
    block_2 = layer_fn(
        block_args['out_channels'],
        block_args['out_channels'],
        kernel_size = 3,
        stride      = 1,
        padding     = 1
    )

    layer_list = list(block_1) + [block_2]
    return nn.Sequential(*layer_list)


# TPC data compression project-specific blocks
class TPCResidualBlock(nn.Module):
    """
    A residual block that has a double block on the main path
    and a single block on the side path.
    """
    def __init__(
        self,
        main_block,
        side_block,
        activ  = None,
        norm   = None,
        rezero = True
    ):
        """
        Input:
            - main_block (nn.Module): the network block on the main path
            - side_block (nn.Module): the network block on the side path
            - activ (str or dictionary): config of activation
            - norm (str): type of normalization
        Output:
        """
        super().__init__()
        assert main_block[0].in_channels == side_block[0].in_channels, \
            ('main-path block and side-path block'
             'must have the same in_channels')
        assert main_block[0].out_channels == side_block[0].out_channels, \
            ('main-path block and side-path block'
             'must have the same out_channels')

        self.main_block = main_block
        self.side_block = side_block

        out_channels = main_block[0].out_channels

        self.activ = get_activ_layer(activ)
        norm_fn    = get_norm_layer_fn(norm)
        self.norm  = norm_fn(out_channels)

        if rezero:
            self.rezero_alpha = nn.Parameter(torch.zeros((1, )))
        else:
            self.rezero_alpha = 1

    def forward(self, x_input):
        """
        forward
        """
        x_side   = self.side_block(x_input)
        x_main   = self.main_block(x_input)
        x_output = self.rezero_alpha * x_main + x_side
        return self.norm(self.activ(x_output))


def encoder_residual_block(conv_args, activ=None, norm=None, rezero=True):
    """
    Get an encoder residual block.
    """
    return TPCResidualBlock(
        main_block = double_block('conv', conv_args, activ, norm),
        side_block = single_block('conv', conv_args, activ, norm),
        activ      = activ,
        norm       = norm,
        rezero     = rezero
    )

def decoder_residual_block(deconv_args, activ=None, norm=None, rezero=True):
    """
    Get an decoder residual block.
    """
    return TPCResidualBlock(
        main_block = double_block('deconv', deconv_args, activ, norm),
        side_block = single_block('deconv', deconv_args, activ, norm),
        activ      = activ,
        norm       = norm,
        rezero     = rezero
    )

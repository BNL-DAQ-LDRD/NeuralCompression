"""
yhuang2@bnl.gov
"""

import torch.nn as nn
from neuralcompress.models.cae import Encoder, Decoder


# The Bicephalous Auto-Encoder
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
        self.decoder_c.set_half_mode(False)
        self.decoder_r.set_half_mode(False)

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

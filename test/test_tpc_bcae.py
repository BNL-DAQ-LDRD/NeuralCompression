"""
Test the `neuralcompress/models/blocks.py`
"""
from neuralcompress.models.bcae import BCAE
import torch


if __name__ == '__main__':

    # Construct encoder network
    conv_layer_1 = {
        'out_channels': 8,
        'kernel_size' : [4, 3, 3],
        'padding'     : [1, 0, 1],
        'stride'      : [2, 2, 1]
    }
    conv_layer_2 = {
        'out_channels': 16,
        'kernel_size' : [4, 4, 3],
        'padding'     : [1, 1, 1],
        'stride'      : [2, 2, 1]
    }
    conv_layer_3 = {
        'out_channels': 32,
        'kernel_size' : [4, 4, 3],
        'padding'     : [1, 1, 1],
        'stride'      : [2, 2, 1]
    }
    conv_layer_4 = {
        'out_channels': 32,
        'kernel_size' : [4, 3, 3],
        'padding'     : [1, 0, 1],
        'stride'      : [2, 2, 1]
    }

    # Construct decoder network
    deconv_layer_1 = {
        'out_channels': 8,
        'kernel_size' : [4, 3, 3],
        'padding'     : [1, 0, 1],
        'stride'      : [2, 2, 1],
        'output_padding': 0
    }
    deconv_layer_2 = {
        'out_channels': 4,
        'kernel_size' : [4, 4, 3],
        'padding'     : [1, 1, 1],
        'stride'      : [2, 2, 1],
        'output_padding': 0
    }
    deconv_layer_3 = {
        'out_channels': 2,
        'kernel_size' : [4, 4, 3],
        'padding'     : [1, 1, 1],
        'stride'      : [2, 2, 1],
        'output_padding': 0
    }
    deconv_layer_4 = {
        'out_channels': 1,
        'kernel_size' : [4, 3, 3],
        'padding'     : [1, 0, 1],
        'stride'      : [2, 2, 1],
        'output_padding': 0
    }
    conv_args_list = [conv_layer_1, conv_layer_2,
                      conv_layer_3, conv_layer_4]
    deconv_args_list = [deconv_layer_1, deconv_layer_2,
                        deconv_layer_3, deconv_layer_4]
    bcae = BCAE(
        image_channels   = 1,
        code_channels    = 8,
        conv_args_list   = conv_args_list,
        deconv_args_list = deconv_args_list,
        activ            = {'name': 'leakyrelu', 'negative_slope': .1},
        norm             = 'instance'
    )
    print(bcae)

    # Get number of parameters
    total_numel = 0
    for parameter in bcae.parameters():
        total_numel += parameter.numel()
    print(f'{total_numel/1e3:.2f}K')

    # Encoding
    data = torch.randn(32, 1, 192, 249, 16)
    decom_clf, decom_reg = bcae(data)
    print(decom_clf.shape)

    # script and save
    random_input = torch.randn(1, 1, 192, 249, 16, dtype=torch.float32)
    traced = torch.jit.trace(bcae, random_input)
    traced.save('checkpoints/test_tpc_bcae/bcae.pt')

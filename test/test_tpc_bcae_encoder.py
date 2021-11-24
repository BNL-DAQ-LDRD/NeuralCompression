"""
Test the `neuralcompress/models/bcae.py`
"""
from neuralcompress.models.bcae import (
    Encoder,
    Decoder,
    BCAE
)
import torch


if __name__ == '__main__':

    layer_1 = {
        'out_channels': 8,
        'kernel_size' : [4, 3, 3],
        'padding'     : [1, 0, 1],
        'stride'      : [2, 2, 1]
    }
    layer_2 = {
        'out_channels': 16,
        'kernel_size' : [4, 4, 3],
        'padding'     : [1, 1, 1],
        'stride'      : [2, 2, 1]
    }
    layer_3 = {
        'out_channels': 32,
        'kernel_size' : [4, 4, 3],
        'padding'     : [1, 1, 1],
        'stride'      : [2, 2, 1]
    }
    layer_4 = {
        'out_channels': 32,
        'kernel_size' : [4, 3, 3],
        'padding'     : [1, 0, 1],
        'stride'      : [2, 2, 1]
    }


    conv_args_list = [layer_1, layer_2, layer_3, layer_4]

    encoder = Encoder(
        input_channels  = 1,
        conv_args_list  = conv_args_list,
        activ           = {'name': 'leakyrelu', 'negative_slope': .1},
        norm            = 'instance',
        output_channels = 8
    )
    print(encoder)
    total_numel = 0
    for parameter in encoder.parameters():
        total_numel += parameter.numel()
    print(f'{total_numel/1e6:.2f}M')

    tensor_in = torch.randn(32, 1, 192, 249, 16)
    tensor_out = encoder(tensor_in)
    print(tensor_out.shape)

    scripted = torch.jit.script(encoder)
    scripted.save('checkpoints/test_tpc_bcae/encoder.pt')

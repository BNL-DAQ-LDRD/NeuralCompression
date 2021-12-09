"""
Test the Encoder class in `neuralcompress/models/bcae.py`
"""
from neuralcompress.models.bcae import Encoder
import torch


if __name__ == '__main__':

    # Construct encoder network
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
    encoder = Encoder(
        input_channels  = 1,
        conv_args_list  = [layer_1, layer_2, layer_3, layer_4],
        activ           = {'name': 'leakyrelu', 'negative_slope': .1},
        norm            = 'instance',
        output_channels = 8
    )
    print(encoder)

    # Get number of parameters
    total_numel = 0
    for parameter in encoder.parameters():
        total_numel += parameter.numel()
    print(f'{total_numel/1e3:.2f}K')

    # Encoding
    tensor_in = torch.randn(32, 1, 192, 249, 16)
    tensor_out = encoder(tensor_in)
    print(tensor_out.shape)

    # script and save
    scripted = torch.jit.script(encoder)
    scripted.save('checkpoints/test_tpc_bcae/encoder.pt')

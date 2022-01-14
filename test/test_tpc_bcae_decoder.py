"""
Test the Decoder class in `neuralcompress/models/bcae.py`
"""
from neuralcompress.models.bcae import Decoder
import torch


if __name__ == '__main__':

    # Construct decoder network
    layer_1 = {
        'out_channels': 8,
        'kernel_size' : [4, 3, 3],
        'padding'     : [1, 0, 1],
        'stride'      : [2, 2, 1],
        'output_padding': 0
    }
    layer_2 = {
        'out_channels': 4,
        'kernel_size' : [4, 4, 3],
        'padding'     : [1, 1, 1],
        'stride'      : [2, 2, 1],
        'output_padding': 0
    }
    layer_3 = {
        'out_channels': 2,
        'kernel_size' : [4, 4, 3],
        'padding'     : [1, 1, 1],
        'stride'      : [2, 2, 1],
        'output_padding': 0
    }
    layer_4 = {
        'out_channels': 1,
        'kernel_size' : [4, 3, 3],
        'padding'     : [1, 0, 1],
        'stride'      : [2, 2, 1],
        'output_padding': 0
    }
    decoder = Decoder(
        input_channels   = 8,
        deconv_args_list = [layer_1, layer_2, layer_3, layer_4],
        activ            = {'name': 'leakyrelu', 'negative_slope': .1},
        norm             = 'instance',
        output_channels  = 1,
        output_activ     = None,
        output_norm      = None
    )
    print(decoder)

    # Get number of parameters
    total_numel = 0
    for parameter in decoder.parameters():
        total_numel += parameter.numel()
    print(f'{total_numel/1e3:.2f}K')

    # Decoding
    tensor_in = torch.randn(32, 8, 12, 15, 16)
    tensor_out = decoder(tensor_in)
    print(tensor_out.shape)

    # Trace and save
    # Note that scripts doesn't work with amp.autocast() but trace does.

    random_input = torch.randn(1, 8, 12, 15, 16, dtype=torch.float32)
    decoder_traced = torch.jit.trace(decoder, random_input)
    decoder_traced.save('checkpoints/test_tpc_bcae/decoder.pt')

"""
Test the Encoder class in `neuralcompress/models/bcae.py`
"""
from neuralcompress.utils.tpc_dataloader import get_tpc_dataloaders
from neuralcompress.models.bcae import Encoder
import torch
import sys
from time import time


if __name__ == '__main__':
    
    config = {
        'data_path': '/data/datasets/sphenix/highest_framedata_3d/outer',
        'data': {
            'batch_size' : int(sys.argv[1]),
            'train_sz'   : 2560,
            'valid_sz'   : 0,
            'test_sz'    : 0,
            'is_random'  : True,
        }
    }
    device = sys.argv[2]
    assert device in ['cpu', 'cuda']

    train_loader, _, _ = get_tpc_dataloaders(
        config['data_path'],
        **config['data']
    )
    

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
    encoder.to(device)

    # Get number of parameters
    total_numel = 0
    for parameter in encoder.parameters():
        total_numel += parameter.numel()
    print(f'{total_numel/1e3:.2f}K')

    # warmup
    for batch in train_loader:
        batch = batch.to(device)
        tensor_out = encoder(batch)
    
    # Encoding
    T = 10
    time0 = time()
    with torch.no_grad():
        for _ in range(T):
            for batch in train_loader:
                tensor_out = encoder(batch)
    time_total = time() - time0
    
    print(f'{time_total:.4f}')
    time_per_input = time_total / (T * config['data']['train_sz'])
    print(f'{time_per_input:.6f}')
    time_per_frame = time_per_input * 24
    print(f'{time_per_frame:.6f}')


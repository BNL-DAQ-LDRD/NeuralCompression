"""
Script the TPC data
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from neuralcompress.utils.tpc_dataloader import get_tpc_dataloaders

DATA_ROOT = '/data/datasets/sphenix/highest_framedata_3d/outer'


# pylint: disable=too-few-public-methods
class TensorContainer(nn.Module):
    """
    A container for tensor.
    Save data for using by C++.
    """
    def __init__(self, tensor_dict):
        super().__init__()
        for key, value in tensor_dict.items():
            setattr(self, key, value)


def main():
    """
    main
    """
    parser = argparse.ArgumentParser(
        description="Script the TPC data"
    )

    parser.add_argument(
        '--data_path',
        required = False,
        default  = DATA_ROOT,
        type     = str,
        help     = "The path to data."
    )

    parser.add_argument(
        '--save_path',
        required = True,
        type     = str,
        help     = "The path to save the scripted data."
    )

    parser.add_argument(
        '--filename',
        required = False,
        default  = 'tpc_data',
        type     = str,
        help     = "The filename of the scripted data | default='tpc_data'."
    )

    parser.add_argument(
        '--data_size',
        required = False,
        default  = 1,
        type     = int,
        help     = "Number of frames to load | default=1."
    )

    parser.add_argument(
        '--partition',
        required = False,
        default  = 'test',
        choices=['train', 'valid', 'test'],
        type     = str,
        help     = "partition from which to load the data | default=test."
    )

    parser.add_argument(
        '--random',
        action = 'store_true',
        help   = "Whether to get a random sample."
    )

    args = parser.parse_args()

    data_config = {
        'batch_size' : args.data_size,
        'train_sz'   : 0,
        'valid_sz'   : 0,
        'test_sz'    : 0,
        'is_random'  : args.random,
    }
    partition = args.partition
    data_config[f'{partition}_sz'] = args.data_size

    data_path = Path(args.data_path)
    assert data_path.exists(), f'{data_path} does not exist!'

    loaders = get_tpc_dataloaders(data_path, **data_config)
    if partition == 'train':
        loader = loaders[0]
    elif partition == 'valid':
        loader = loaders[1]
    else:
        loader = loaders[2]

    data = next(iter(loader))
    print(f'shape of tpc data = {data.shape}')

    container = TensorContainer({'data': data})
    container_traced = torch.jit.script(container)

    filename = args.filename
    if not filename.endswith('.pt'):
        filename += '.pt'

    save_path = Path(args.save_path)
    if not save_path.exists():
        save_path.mkdir(parents=True)

    scripted_filename = save_path/filename
    container_traced.save(scripted_filename)

    print(f'scripted tpc data saved to: {scripted_filename}')


if __name__ == '__main__':
    main()

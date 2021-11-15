"""
Goal:
    Given a dataset API and a split filename, get one or more dataloaders.
"""

#! /usr/bin/env python
import torch
from torch.utils.data import DataLoader, Subset


def get_dataloader(
    dataset_api,
    split_fname,
    batch_size,
    lengths=None,
    seed=None
):
    """
    Take a dataset_api that only have split file return data loaders.
    Input:
        - dataset_api (Dataset API): a dataset api that takes only one input
            which is the split file that contains the list of
            symlinks to the datafile;
        - split_fname (str): The filename of the split file;
        - batch_size (int): batch_size of the data loader;
        - lengths (int or list(int)): if lengths is an integer,
            return a data loader of containing lengths many data points;
            If lengths is a list of integers, return a list of data loaders
            of length len(lengths) where data loaders[i] contains length[i]
            data points;
        - seed (int): If seed is given randomly shuffle the datafiles
    Output:
        - a DataLoader object or a list of DataLoader object.
    """

    # get dataset
    dataset = dataset_api(split_fname)

    # If length is not given return the data loader
    if not lengths:
        return DataLoader(dataset, batch_size=batch_size)

    # If length is given, parse length, and check validity
    if isinstance(lengths, int):
        total_length = lengths
    else:
        total_length = sum(lengths)
    assert total_length <= len(dataset), 'sum(lengths) > len(dataset)'

    # get indices
    # If seed is given, random shuffle the indices
    if seed is not None:
        indices = torch.randperm(
            len(dataset),
            generator=torch.Generator().manual_seed(seed)
        )
        indices = indices[:total_length]
    else:
        indices = torch.arange(0, total_length)

    # if lengths is just one integer, return the dataloader
    if isinstance(lengths, int):
        return DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size
        )
    # if lengths is a sequence, return a sequence of datalaoder
    data_loaders = []
    start = 0
    for length in lengths:
        end = start + length
        loader = DataLoader(
            Subset(dataset, indices[start: end]),
            batch_size=batch_size
        )
        data_loaders.append(loader)
        start = end
    return data_loaders


# def test():
#     """
#     Test with TPC dataset API.
#     """
#     import sys
#     sys.path.append('/home/yhuang2/PROJs/NeuralCompression/neuralcompress/')
#     from pathlib import Path
#     from datasets.tpc_dataset import DatasetTPC3d
#
#     batch_size = 32
#     root_folder = Path('/data/datasets/sphenix/highest_framedata_3d/outer/')
#     train_loader, valid_loader = get_dataloader(
#         DatasetTPC3d,
#         root_folder/'train.txt',
#         batch_size,
#         lengths=[300, 100],
#         seed=0
#     )
#
#     print('\ntrain')
#     for i, batch in enumerate(train_loader):
#         print(f'{i + 1}/{len(train_loader)}: {len(batch)}')
#     print('\nvalid')
#     for i, batch in enumerate(valid_loader):
#         print(f'{i + 1}/{len(valid_loader)}: {len(batch)}')
#
#     loader = get_dataloader(
#         DatasetTPC3d,
#         root_folder/'test.txt',
#         batch_size=32,
#         lengths=200
#     )
#     print('\ntest')
#     for i, batch in enumerate(loader):
#         print(f'{i + 1}/{len(loader)}: {len(batch)}')


if __name__ == "__main__":
    print('This is main of data_loader.py')
    # test()

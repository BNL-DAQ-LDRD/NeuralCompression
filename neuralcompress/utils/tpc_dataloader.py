"""
Get train, valid, and test dataloaders
"""
#! /usr/bin/env python
from pathlib import Path
import numpy as np
from neuralcompress.datasets.tpc_dataset import DatasetTPC3d
import torch
from torch.utils.data import (Subset, DataLoader)

def subsample_dataset(
    dataset,
    sample_sz = None,
    is_random = True,
    seed      = None
):
    """
    subsample a dataset
    """
    if sample_sz is None:
        sample_sz = len(dataset)
    assert 0 <= sample_sz <= len(dataset), \
        f'dataset does not contains sample_sz({sample_sz}) many examples'

    indices = np.arange(len(dataset))
    if is_random:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)

    return Subset(dataset, indices[:sample_sz])


# pylint: disable=too-many-arguments
def get_tpc_test_dataloader(
    manifest,
    batch_size,
    test_sz         = None,
    is_random       = True,
    seed            = None,
    # frequently used DataLoader parameters
    # the default values are set to match those in PyTorch docs.
    shuffle         = False,
    num_workers     = 0,
    pin_memory      = False,
    prefetch_factor = 2,
):
    """
    Get TPC test dataloader
    """
    dataset = subsample_dataset(
        DatasetTPC3d(manifest),
        sample_sz = test_sz,
        is_random = is_random,
        seed      = seed
    )

    return DataLoader(
        dataset,
        batch_size      = batch_size,
        shuffle         = shuffle,
        num_workers     = num_workers,
        pin_memory      = pin_memory,
        prefetch_factor = prefetch_factor
    )

# pylint: disable=too-many-arguments
def get_tpc_train_valid_dataloaders(
    train_manifest,
    batch_size,
    train_sz        = None,
    valid_sz        = None,
    valid_ratio     = None,
    is_random       = True,
    seed            = None,
    # frequently used DataLoader parameters
    # the default values are set to match those in PyTorch docs.
    shuffle         = False,
    num_workers     = 0,
    pin_memory      = False,
    prefetch_factor = 2
):
    """
    Get TPC train and valid dataloaders
    """

    assert (
        (train_sz is not None and valid_sz is not None) or
        (train_sz is None and valid_sz is None and valid_ratio is not None)
    ), 'give train size and valid size or just valid ratio'

    dataset = DatasetTPC3d(train_manifest)

    if valid_ratio is not None:
        train_sz = int(len(dataset) / (1 + valid_ratio))
        valid_sz = len(dataset) - train_sz

    dataset = subsample_dataset(
        dataset,
        sample_sz = train_sz + valid_sz,
        is_random = is_random,
        seed      = seed
    )
    train_dataset = Subset(dataset, torch.arange(0, train_sz))
    valid_dataset = Subset(dataset, torch.arange(train_sz, len(dataset)))

    train_loader = DataLoader(
        train_dataset,
        batch_size      = batch_size,
        shuffle         = shuffle,
        num_workers     = num_workers,
        pin_memory      = pin_memory,
        prefetch_factor = prefetch_factor
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size      =batch_size,
        shuffle         = shuffle,
        num_workers     = num_workers,
        pin_memory      = pin_memory,
        prefetch_factor = prefetch_factor
    )
    return train_loader, valid_loader


# pylint: disable=too-many-arguments
def get_tpc_dataloaders(
    manifest_path,
    batch_size,
    train_sz        = None,
    valid_sz        = None,
    valid_ratio     = None,
    test_sz         = None,
    is_random       = True,
    seed            = None,
    # frequently used DataLoader parameters
    # the default values are set to match those in PyTorch docs.
    # If shuffle is set to true,
    # reshuffle the data at every epoch
    shuffle         = False,
    # Number of subprocess to use for data loading.
    # 0 means that the data will be loaded in the main process.
    # Set a positive number to enable multi-process data loading.
    num_workers     = 0,
    # If True, the data loader will copy tensors into CIDA pinned
    # memory before returning them.
    # This will speed up data transfer to CUDA-enabled GPUs.
    pin_memory      = False,
    #  Number of samples loaded in advance by each worker.
    # 2 means there will be a total of 2 * num_workers
    # samples prefetched across all workers. (default: 2)
    prefetch_factor = 2,
):
    """
    Get TPC train, valid, and test dataloaders
    """
    test_manifest = Path(manifest_path)/'test.txt'
    assert test_manifest.exists(), \
        f'{test_manifest} does not exist.'
    test_loader = get_tpc_test_dataloader(
        test_manifest,
        batch_size,
        test_sz         = test_sz,
        is_random       = is_random,
        seed            = seed,
        shuffle         = shuffle,
        num_workers     = num_workers,
        pin_memory      = pin_memory,
        prefetch_factor = prefetch_factor
    )

    train_manifest = Path(manifest_path)/'train.txt'
    assert test_manifest.exists(), \
        f'{train_manifest} does not exist.'
    train_loader, valid_loader = get_tpc_train_valid_dataloaders(
        train_manifest,
        batch_size,
        train_sz        = train_sz,
        valid_sz        = valid_sz,
        valid_ratio     = valid_ratio,
        is_random       = is_random,
        seed            = seed,
        shuffle         = shuffle,
        num_workers     = num_workers,
        pin_memory      = pin_memory,
        prefetch_factor = prefetch_factor
    )
    return train_loader, valid_loader, test_loader

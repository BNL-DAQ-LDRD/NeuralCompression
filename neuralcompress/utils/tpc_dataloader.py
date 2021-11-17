"""
"""
#! /usr/bin/env python
import torch
from torch.utils.data import DataLoader, Subset
from neuralcompress.dataset.tpc_dataset import DatasetTPC3d
from neuralcompress.utils.dataset_utils import (
    sample_dataset,
    split_dataset
)

def get_tpc_test_dataloader(
    test_manifest,
    batch_size,
    length=None,
    shuffle=True,
    seed=None
):
    """
    """
    dataset = DatasetTPC3d(test_manifest)
    dataset = sample_dataset(dataset, length, shuffle, seed)
    return DataLoader(dataset, batch_size=batch_size)


def get_tpc_train_valid_dataloaders(
    train_manifest,
    batch_size,
    train_length=None,
    valid_length=None,
    valid_ratio=None,
    shuffle=True,
    seed=None
):
    """
    """
    assert (
        (train_length != None and valid_length != None) and
        (train_length == None and valid_length == None) and valid_ratio != None
    ), f'give train length and valid_length or just valid_ratio'

    dataset = DatasetTPC3d(train_manifest)

    if train_length is not None:
        train_dataset, valid_dataset = split_dataset(
            dataset,
            lengths=[train_length, valid_length],
            shuffle=shuffle,
            seed=seed
        )
    else:
        train_dataset, valid_dataset = split_dataset(
            dataset,
            fractions=[1, valid_ratio],
            shuffle=shuffle,
            seed=seed
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    return train_loader, valid_loader


def get_tpc_dataloaders(
    manifest_path,
    batch_size,
    train_length=None,
    valid_length=None,
    valid_ratio=None,
    test_length=None,
    shuffle=True,
    seed=None
):
    test_manifest = Path(manifest_path)/'test.txt'
    test_loader = get_tpc_test_loader(
        test_manifest,
        batch_size,
        test_length,
        shuffle,
        seed)
    train_manifest = Path(manifest_path)/'train.txt'
    train_loader, valid_loader = get_tpc_train_valid_loaders(
        train_manifest,
        batch_size,
        train_length,
        valid_length,
        valid_ratio,
        shuffle,
        seed
    )
    return train_loader, valid_loader, test_loader

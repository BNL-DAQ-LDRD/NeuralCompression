"""
Get TPC train, valid, and test dataloaders
"""
#! /usr/bin/env python
from pathlib import Path
from torch.utils.data import DataLoader
from neuralcompress.datasets.tpc_dataset import DatasetTPC3d
from neuralcompress.utils.dataset_utils import (
    sample_dataset,
    split_dataset
)

def get_tpc_test_dataloader(
    test_manifest,
    batch_size,
    test_sz=None,
    shuffle=True,
    seed=None
):
    """
    Get TPC test dataloader
    """
    dataset = DatasetTPC3d(test_manifest)
    dataset = sample_dataset(dataset, test_sz, shuffle, seed)
    return DataLoader(dataset, batch_size=batch_size)

# pylint: disable=too-many-arguments
def get_tpc_train_valid_dataloaders(
    train_manifest,
    batch_size,
    train_sz=None,
    valid_sz=None,
    valid_ratio=None,
    shuffle=True,
    seed=None
):
    """
    Get TPC train and valid dataloaders
    """

    assert (
        (train_sz is not None and valid_sz is not None) or
        (train_sz is None and valid_sz is None and valid_ratio is not None)
    ), 'give train size and valid size or just valid ratio'

    dataset = DatasetTPC3d(train_manifest)

    if train_sz is not None:
        train_dataset, valid_dataset = split_dataset(
            dataset,
            sizes=[train_sz, valid_sz],
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


# pylint: disable=too-many-arguments
def get_tpc_dataloaders(
    manifest_path,
    batch_size,
    train_sz=None,
    valid_sz=None,
    valid_ratio=None,
    test_sz=None,
    shuffle=True,
    seed=None
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
        test_sz=test_sz,
        shuffle=shuffle,
        seed=seed
    )

    train_manifest = Path(manifest_path)/'train.txt'
    assert test_manifest.exists(), \
        f'{train_manifest} does not exist.'
    train_loader, valid_loader = get_tpc_train_valid_dataloaders(
        train_manifest,
        batch_size,
        train_sz=train_sz,
        valid_sz=valid_sz,
        valid_ratio=valid_ratio,
        shuffle=shuffle,
        seed=seed
    )
    return train_loader, valid_loader, test_loader

"""
Get TPC train, valid, and test dataloaders
"""
#! /usr/bin/env python
from pathlib import Path
from torch.utils.data import DataLoader
from datasets.tpc_dataset import DatasetTPC3d
from utils.dataset_utils import (
    sample_dataset,
    split_dataset
)

def get_tpc_test_dataloader(
    test_manifest,
    batch_size,
    sample_sz=None,
    shuffle=True,
    seed=None
):
    """
    Get TPC test dataloader
    """
    dataset = DatasetTPC3d(test_manifest)
    dataset = sample_dataset(dataset, sample_sz, shuffle, seed)
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
        (train_sz is not None and valid_sz is not None) and
        (train_sz is None and valid_sz is None) and valid_ratio is not None
    ), 'give train length and valid_sz or just valid_ratio'

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
    test_loader = get_tpc_test_dataloader(
        test_manifest,
        batch_size,
        test_sz,
        shuffle,
        seed)
    train_manifest = Path(manifest_path)/'train.txt'
    train_loader, valid_loader = get_tpc_train_valid_dataloaders(
        train_manifest,
        batch_size,
        train_sz,
        valid_sz,
        valid_ratio,
        shuffle,
        seed
    )
    return train_loader, valid_loader, test_loader

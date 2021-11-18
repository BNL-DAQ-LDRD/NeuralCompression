"""
Sample and split dataset
"""
#! /usr/bin/env python
import torch
from torch.utils.data import Subset
import numpy as np

def sample(
    total_sz,
    sample_sz=None,
    shuffle=True,
    seed=None
):
    """
    Input:
        - total_sz (int): Total sample_sz.
        - sample_sz (int): sample_sz needed.
        - shuffle (boolean): whether to shuffle the indices.
        - seed (int): seed for rng.
    Output:
        sampled indices
    """
    if sample_sz is None:
        sample_sz = total_sz
    else:
        assert 0 <= sample_sz <= total_sz, \
            'sample_sz should be a integer in [0, total_sz]'

    if shuffle:
        if seed is None:
            gen = None
        else:
            gen = torch.Generator().manual_seed(seed)
        indices = torch.randperm(total_sz, generator=gen)
        indices = indices[:sample_sz]
    else:
        indices = torch.arange(0, sample_sz)
    return indices


def sample_dataset(
    dataset,
    sample_sz=None,
    shuffle=True,
    seed=None
):
    """
    Input:
        - dataset (Dataset): A torch Dataset object.
        - sample_sz (int): Upper bound of the number of
            examples in the dataset.
        - shuffle (boolean): whether to shuffle the dataset.
        - seed (int): seed for rng.
    Output:
        sampled dataset
    """
    indices = sample(len(dataset), sample_sz, shuffle, seed)
    return Subset(dataset, indices)


def split(
    total_sz,
    sizes=None,
    fractions=None,
    shuffle=True,
    seed=None,
):
    """
    Input:
        - total_sz (int): Total sample_sz.
        - sizes (sequence of integers):sample_sz of each part.
        - fractions (sequence of numerical values):
            We first normalize fractions by its sum and
            then use the normalized entry as the fraction
            of each part.
        - shuffle (boolean): whether to shuffle the indices.
        - seed (int): seed for rng.
    Output:
        indices in each part
    """

    # get cumulative sample_sz
    assert (sizes is None) ^ (fractions is None), \
        'exactly one of sizes and fractions must be given'

    if sizes:
        assert all(0 <= size <= total_sz for size in sizes)
        cumulative_sizes = np.cumsum(sizes)
        assert cumulative_sizes[-1] <= total_sz, \
            'sum of sizes is more than total_sz'
    else:
        assert all(fraction >= 0 for fraction in fractions)
        cumulative_fractions = np.cumsum(fractions)
        cumulative_fractions /= cumulative_fractions[-1]
        cumulative_sizes = map(int, cumulative_fractions * total_sz)

    indices = sample(total_sz, cumulative_sizes[-1], shuffle, seed)

    # get indices for each part
    start, indices_list = 0, []
    for end in cumulative_sizes:
        indices_list.append(indices[start: end])
        start = end

    return indices_list


def split_dataset(
    dataset,
    sizes=None,
    fractions=None,
    shuffle=True,
    seed=None
):
    """
    Input:
        - dataset (Dataset): A torch Dataset object.
        - sizes (sequence of integer):
            number of examples in each sub-dataset.
        - fractions (sequence of numerical values):
            We first normalize fractions by its sum and
            then use the normalized entry as the fraction
            of examples in each sub-dataset.
        - shuffle (boolean): whether to shuffle the dataset
            if sample_sz is given and <= len(dataset).
        - seed (int): seed for rng.
    Output:
        sub-datasets
    """

    indices_list = split(len(dataset), sizes, fractions, shuffle, seed)
    return [Subset(dataset, indices) for indices in indices_list]

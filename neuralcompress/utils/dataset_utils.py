"""
Sample and split dataset
"""
#! /usr/bin/env python
import torch
from torch.utils.data import Subset

def sample(
    total,
    length=None,
    shuffle=True,
    seed=None
):
    """
    Input:
        - total (int): Total length.
        - length (int): length needed.
        - shuffle (boolean): whether to shuffle the indices.
        - seed (int): seed for rng.
    """
    if length is None:
        length = total
    else:
        length = min(total, length)

    if shuffle:
        gen = torch.Generator().manual_seed(seed)
        indices = torch.randperm(total, generator=gen)
        indices = indices[:length]
    else:
        indices = torch.arange(0, length)
    return indices


def sample_dataset(
    dataset,
    length=None,
    shuffle=True,
    seed=None
):
    """
    Input:
        - dataset (Dataset): A torch Dataset object.
        - length (int): Upper bound of the number of examples in the dataset.
        - shuffle (boolean): whether to shuffle the dataset.
        - seed (int): seed for rng.
    """
    indices = sample(len(dataset), length, shuffle, seed)
    return Subset(dataset, indices),


def split(
    total,
    lengths=None,
    fractions=None,
    shuffle=True,
    seed=None,
):
    """
    Input:
        - total (int): Total length.
        - lengths (sequence of integers):length of each part.
        - fractions (sequence of numerical values):
            We first normalize fractions by its sum and
            then use the normalized entry as the fraction
            of each part.
        - shuffle (boolean): whether to shuffle the indices.
        - seed (int): seed for rng.
    """

    # get cumulative length
    assert (lengths != None) ^ (fractions != None), \
        f'exactly one of lengths and fractions must be given'

    if lengths:
        cumulative_length = np.cumsum(lengths)
        assert cumulative_length[-1] <= total, \
            f'sum of lengths is more than total'
    else:
        cumulative_fraction = np.cumsum(fractions)
        cumulative_length = map(int, cumulative_fraction * total)

    indices = sample(len(dataset), length, shuffle, seed)

    # get indices for each part
    start, indices_list = 0, []
    for end in cumulative_length:
        indices_list.append(indices[start: end])
        start = end

    return indices_list


def split_dataset(
    dataset,
    lengths=None,
    fractions=None,
    shuffle=True
    seed=None
):
    """
    Input:
        - dataset (Dataset): A torch Dataset object.
        - lengths (sequence of integer):
            number of examples in each sub-dataset.
        - fractions (sequence of numerical values):
            We first normalize fractions by its sum and
            then use the normalized entry as the fraction
            of examples in each sub-dataset.
        - shuffle (boolean): whether to shuffle the dataset
            if length is given and <= len(dataset).
        - seed (int): seed for rng.
    """

    indices_list = split(len(dataset), lengths, fractions, shuffle, seed)
    return [Subset(dataset, indices) for indices in indices_list]

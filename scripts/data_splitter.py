"""
Goal:
    Generate split files.
Note:
    This script is supposed to be run once and generates
    two split files train.txt and test.txt.
Usage Examples:
    1. python data_splitter.py -d [path/to/data] --train 1 --test 1 -r 0
        Recursively search all npy files in [path/to/data],
        shuffle the files with rng with random state r=0,
        split the data files into train and test with ratio=1:1, and
        save the train data file paths to [path/to/data]/train.txt and
        the test data file paths to [path/to/data]/text.txt.
    2. python data_splitter.py -d [path/to/data] -s [split/file/path]
        Use the default train-test ratio = 8:2.
        The filenames are not shuffled.
        Save the train data file paths to [split/file/path]/train.txt and
        the test data file paths to [split/file/path]/text.txt
    3. python data_splitter.py -d [path/to/data] --train 3 --valid 2 --test 1
        Use train valid test ratio = 3:2:1.
"""

#!/usr/bin/env python

import argparse
import os
from pathlib import Path
import numpy as np

def split(
    data_path,
    split_path=None,
    ratios=None,
    rng_random_state=None,
):
    """
    - data_path: input dir that contains npy files
        or subfolders (work with data_path_glob)
        that contains npy files.

    - split_path: output dir that contains the splits
        (train.txt, valid.txt, and text.txt).

    - shuffle: whether to shuffle
        the data files before splitting.
        Default is True.

    - rng_random_state: seed for numpy random number generator.
        Default is None.

    - ratios: A dictionary with split name as key and ratio as value.
        Default is {'train': 8, 'test': 2}.

    """

    data_path = Path(data_path)
    assert data_path.is_dir(),\
        f'{data_path} does not exist.'

    file_list = sorted(list(data_path.rglob('*npy')))
    assert len(file_list) > 0, \
        f'{data_path} and its subdirectories \
        do not contain any npy files'

    if rng_random_state is not None:
        rng = np.random.RandomState(rng_random_state)
        rng.shuffle(file_list)

    if split_path:
        split_path = Path(split_path)
        split_path.mkdir(parents=True, exist_ok=True)
    else:
        split_path = data_path

    total = 0
    for ratio in ratios.values():
        total += ratio

    if not ratios:
        ratios = {'train': 8, 'test': 2}

    start, psum = 0, 0
    for _split, ratio in ratios.items():
        psum += ratio
        end = int(len(file_list) * psum / total)
        create_subset(
            split_path,
            file_list,
            _split,
            range(start, end)
        )
        start = end

def create_subset(split_path, file_list, name, list_range):
    """
    create split file with symbolic link
    """

    out_file = Path(split_path)/(name + '.txt')

    with open(out_file, 'w') as file_handle:
        for i in list_range:
            filename = file_list[i]
            file_handle.write(str(filename.resolve()))
            file_handle.write(os.linesep)


def main():
    """
    Parse command-line argument and run splitter.
    """
    parser = argparse.ArgumentParser(
        description='Split Data to train, valid and test')

    parser.add_argument(
        '-d',
        '--data_path',
        type=str,
        required=True,
        help='Input dir that contains the npy data files.\
            Note that the input dir will be \
            searched recursively for data files'
    )
    parser.add_argument(
        '-s',
        '--split_path',
        type=str,
        default=None,
        help='Output dir that stores split files.'
    )
    parser.add_argument(
        '-r',
        '--rng_random_state',
        type=int,
        default=None,
        help='Random state for the numpy random number generator.'
    )
    parser.add_argument(
        '--train',
        type=int,
        default=8,
        help='train ratio'
    )
    parser.add_argument(
        '--valid',
        type=int,
        default=0,
        help='train ratio'
    )
    parser.add_argument(
        '--test',
        type=int,
        default=2,
        help='test ratio'
    )
    args = vars(parser.parse_args())

    ratios = {}
    for _split in ['train', 'valid', 'test']:
        if args[_split] > 0:
            ratios[_split] = args[_split]

    split(
        data_path        = args['data_path'],
        split_path       = args['split_path'],
        rng_random_state = args['rng_random_state'],
        ratios           = ratios,
    )


if __name__ == "__main__":
    main()

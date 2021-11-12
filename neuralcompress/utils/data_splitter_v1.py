#!/usr/bin/env python

import argparse
import os
from pathlib import Path
import numpy as np

def create_subset(output_dir, file_list, name, list_range):

    """
    **Data Splitting**
    Split the data into `train`, `valididation` (`valid`), and `test`.
    create subfolder and symbolic link
    """

    assert name in ('train', 'valid', 'test')
    outfile = Path(output_dir)/(name + '.txt')

    with open(outfile, 'w') as fp:
        for i in list_range:
            fn = file_list[i]
            fp.write(str(fn.resolve()))
            fp.write(os.linesep)


def split(
    input_dir,
    output_dir,
    input_dir_glob=None,
    max_examples=None,
    ratios=(8, 1, 1),
    rnd_seed=None):

    """
    Split the data into 3 parts: train, validation (valid), and test.
    **Input**:
        - input_dir: input dir that contains npy files
            or subfolders (work with input_dir_glob) that contains npy files.
        - output_dir: output dir that contains the splits (train.txt, valid.txt, and text.txt)
            or subfolders (work with input_dir_glob) that contains the splits.
        - input_dir_glob (optional): the pattern of subfolders (under input_dir) containing npy files;
            For example, input_dir_glob = "inner/12-2_*/" means looking for all paths
            with the form [input_dir]/inner/12-2_*/ for npy files.
            Suppose "[input_dir]/inner/12-2_4-0" is such a folder,
            then the corresponding split files are saved to "[output_dir]/inner/12-2_4-0".
        - max_examples (optional): an integer that is the upper bound of the
            maximum number of total examples (train + valid + test) to use.
            If not provided, use all.
        - ratios (optional): 3 integers indicating the train : valid : test ratios.
            For example, suppose we have 10000 npy files, and we use ratios = (8, 1, 1),
            then we will have 8000 training examples, 1000 validation examples, and 1000 test examples.
        - rnd_seed (optional): an integer used for np.random.seed(rnd_seed), default is None.
    """

    input_dir = Path(input_dir)
    assert input_dir.is_dir()

    input_dirs, subdirs = [], []
    if input_dir_glob is not None:
        input_dirs = list(input_dir.glob(input_dir_glob))
        subdirs = [str(d)[len(str(input_dir)):].lstrip('/') for d in input_dirs]
    else:
        input_dirs = [input_dir]
        subdirs = ['']

    output_dir = Path(output_dir)
    output_dirs = [output_dir/subdir for subdir in subdirs]
    for input_dir, output_dir in zip(input_dirs, output_dirs):
        print(f'{input_dir} -> {output_dir}')
    if input('\nContinue?[Y/n]') != 'Y':
        print('terminating')
        exit(1)

    if rnd_seed:
        np.random.seed(rnd_seed)


    for i, (input_dir, output_dir) in enumerate(zip(input_dirs, output_dirs)):
        print(f'[{i + 1}/{len(output_dirs)}] {input_dir}')
        output_dir.mkdir(parents=True, exist_ok=True)

        file_list = sorted(list(input_dir.glob('*.npy')))
        np.random.shuffle(file_list)
        max_examples = len(file_list) if max_examples is None else max_examples
        file_list = file_list[: max_examples]

        sz, total = len(file_list), sum(ratios)
        train_end = int(sz * ratios[0] / total)
        valid_end = int(sz * (ratios[0] + ratios[1]) / total)
        create_subset(output_dir, file_list, 'train', range(train_end))
        create_subset(output_dir, file_list, 'valid', range(train_end, valid_end))
        create_subset(output_dir, file_list, 'test', range(valid_end, sz))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split Data to train, valid, and test')
    parser.add_argument('-i', '--input_dir', type=str, required='True', help='input dir that contains npy files.')
    parser.add_argument('-g', '--input_dir_glob', type=str, default=None, help='input dir glob.')
    parser.add_argument('-o', '--output_dir', type=str, required='True', help='output dir that stores symlinks.')
    parser.add_argument('-r', '--random_seed', type=int, default=None, help='specify the random seed.')
    parser.add_argument('-m', '--max_examples', type=int, default=None, help='the maximum number of examples. If not given, use all.')
    parser.add_argument('-s', '--ratios', nargs=3, type=int, default=[8, 1, 1], help='3 integers, (train : valid : test) ratios.')
    args = parser.parse_args()

    input_dir = args.input_dir
    input_dir_glob = args.input_dir_glob
    output_dir = args.output_dir
    max_examples = args.max_examples
    random_seed = args.random_seed
    ratios = args.ratios

    split(
        input_dir,
        output_dir,
        input_dir_glob=input_dir_glob,
        ratios=ratios,
        max_examples=max_examples,
        random_seed=random_seed
    )


""" Split Data into train, validation and test.
    Data are TPC Frame Data, extracted from h5 using
    `data_extractor.py`

    author: Yihui Ren
    email : yren@bnl.gov
"""

import argparse
import os
from pathlib import Path, PurePath
import numpy as np


def create_subset(name, file_list, list_range, output_dir):
    """ create sub folders and sym link
    """
    assert name in ("train", "valid", "test")
    outfile = output_dir/(name+".txt")
    with open(outfile, 'w') as fp:
        for i in list_range:
            fn = file_list[i]
            fp.write(str(fn.resolve()))
            fp.write(os.linesep)


def main(input_dir, output_dir, rnd_seed):
    """
    Issues:
        - the file_list is not really shuffled;
        - better not to hard code the split ratios;
    Yi has a updated version in Yi's mlgpu account.
    Yi will update later.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    assert input_dir.is_dir() and input_dir.exists()
    output_dir.mkdir(parents=True, exist_ok=True)

    file_list = sorted(list(input_dir.glob("*.npy")))
    sz = len(file_list)
    np.random.seed(rnd_seed)
    idx = np.arange(sz)
    np.random.shuffle(idx)

    # 8:1:1 split
    s, t = int(sz*0.8), int(sz*0.9)
    create_subset("train", file_list, range(0, s), output_dir)
    create_subset("valid", file_list, range(s, t), output_dir)
    create_subset("test", file_list, range(t, sz), output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Split Data to train, valid and test')
    parser.add_argument('-i', '--input_dir', type=str,
                        help='input dir that contains npy files')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='output dir that stores symlinks')
    parser.add_argument('-r', '--random_seed', type=int,
                        help='specify the random seed')
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.random_seed)

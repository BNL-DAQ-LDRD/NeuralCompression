"""
Split Data into train, validation and test.
"""
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
    out_file = Path(output_dir)/(name + '.txt')

    with open(out_file, 'w') as file_handle:
        for i in list_range:
            filename = file_list[i]
            file_handle.write(str(filename.resolve()))
            file_handle.write(os.linesep)


def split(
    input_dir,
    output_dir,
    ratios=(8, 1, 1),
    shuffle=True,
    rng_random_state=None,
    max_sizes=None
):

    """
    Split the data into 3 parts: train, validation (valid), and test.
    **Input**:
        - input_dir: input dir that contains npy files
            or subfolders (work with input_dir_glob) that contains npy files.

        - output_dir: output dir that contains the splits
            (train.txt, valid.txt, and text.txt).

        - shuffle (optional): whether to shuffle
            the data files before splitting.
            Default is True.

        - rnd_random_state (optional): seed for numpy random number generator.
            Default is None.

        - ratios (optional): 3 integers indicating the
            train : valid : test ratios.
            For example, suppose we have 10000 npy files
            and we use ratios = (8, 1, 1),
            then we will have 8000 training examples,
            1000 validation examples, and 1000 test examples.
            Default is (8, 1, 1).

        - max_sizes(optional): if not None, three integers that upper bound
            the number examples in each split.
            Default is None.
    """

    input_dir = Path(input_dir)
    assert input_dir.is_dir(), f'{input_dir} does not exist.'

    file_list = sorted(list(input_dir.rglob('*npy')))
    assert len(file_list) > 0, f'{input_dir} and its subdirectories \
         do not contain any npy files'

    if shuffle:
        rng = np.random.RandomState(rng_random_state)
        rng.shuffle(file_list)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total     = sum(ratios)
    num_files = len(file_list)
    train_end = int(num_files * ratios[0] / total)
    valid_end = int(num_files * (ratios[0] + ratios[1]) / total)

    if max_sizes:
        train_num = min(train_end, max_sizes[0])
        valid_num = min(valid_end - train_end, max_sizes[1])
        test_num  = min(num_files - valid_end, max_sizes[2])
    else:
        train_num = train_end
        valid_num = valid_end - train_end
        test_num  = num_files - valid_end

    create_subset(
        output_dir,
        file_list,
        'train',
        range(train_num)
    )
    create_subset(
        output_dir,
        file_list,
        'valid',
        range(train_end, train_end + valid_num)
    )
    create_subset(
        output_dir,
        file_list,
        'test',
        range(valid_end, valid_end + test_num)
    )


def main():
    """
    Parse command-line argument and run splitter.
    """
    parser = argparse.ArgumentParser(
        description='Split Data to train, valid and test')

    parser.add_argument(
        '-i',
        '--input_dir',
        type=str,
        required=True,
        help='Input dir that contains the npy data files.\
            Note that the input dir will be \
            searched recursively for data files'
    )
    parser.add_argument(
        '-o',
        '--output_dir',
        type=str,
        required=True,
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
        '-s',
        '--shuffle',
        type=bool,
        default=False,
        help='Whether to shuffle the data files before generating the splits.'
    )
    parser.add_argument(
        '-t',
        '--ratios',
        nargs=3,
        type=int,
        default=[8, 1, 1],
        help='Three space-separated integers, (train : valid : test) ratios.'
    )
    parser.add_argument(
        '-m',
        '--max_sizes',
        nargs=3,
        type=int,
        default=None,
        help='Three space-separated integers upper bound the number of \
            examples in the train, validation, and test splits.'
    )
    args = parser.parse_args()

    print(args.shuffle)

    split(
        input_dir        = args.input_dir,
        output_dir       = args.output_dir,
        shuffle          = args.shuffle,
        rng_random_state = args.rng_random_state,
        ratios           = args.ratios,
        max_sizes        = args.max_sizes,
    )


if __name__ == "__main__":
    main()

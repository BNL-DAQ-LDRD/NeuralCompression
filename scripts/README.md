# `data_extractor_crop.py`:
- Goal: cut the entire layer group into non-overlapping sections
    for example: 12 sectors along the azimuth and 2 along the z-direction.

## Usage examples:

1. `python data_extractor_crop.py -i example_h5/ -o example_output/ -m 12 -n 2`
    - `-i`: folder contains the h5 files. the script will simply find ALL files with extension `h5` in the folder.
    - `-o`: the output folder, the folder that will contain the output `npy` files.
    - `-m`: number of sectors along the azimuthal direction.
    - `-n`: number of sections along the z (horizontal) direction.
2. `python data_extractor_crop.py -i example_h5/ -o example_output/ -m 12 -n 2 -g inner`
    - `-g`: layer group(s); we can do one layer group or a subset of ['inner', 'middle', 'outer'].

    If we don't specify `-g`, all layer group will be cropped in the same way.

3. `python data_extractor_crop.py -i example_h5/ -o example_output/ -m 12 -n 2 -g inner -r 3 4 -s 0`
    - `-r`: a integer or a list of integers (space separated); the sectors you want to keep along the azimuthal direction.
    - `-s`: a integer or a list of integers (sapce separated); the sections along the z direction.

    here we want to keep sector 3 and 4 (0-indexed) and the first half along the z direction. Note here we cut the z direction into 2 halves (`-n 2`).
    If `-r` and `-s` are not given, we keep all the sectors and all the sections.

## Structure of the output folder
Suppose the output folder is called `output`, after running the third command above, the structure of the output folder will be
- output
    - inner
        - 12-2_3-0
        - 12-2_4-0

since we only choose inner layer group and sector 3 and 4 and section 0.
The 12-2 means, we have 12 sectors and 2 sections in total.

If we run `python data_extractor_crop.py -i example_h5/ -o example_output/ -m 2 -n 3`, then we will have
- output
    - inner
        - 2-3_0-0
        - 2-3_0-1
        - 2-3_0-2
        - 2-3_1-0
        - 2-3_1-1
        - 2-3_1-2
    - middle
        - 2-3_0-0
        - 2-3_0-1
        - 2-3_0-2
        - 2-3_1-0
        - 2-3_1-1
        - 2-3_1-2
    - outer
        - 2-3_0-0
        - 2-3_0-1
        - 2-3_0-2
        - 2-3_1-0
        - 2-3_1-1
        - 2-3_1-2

# `data_splitter.py`
- Goal: Generate split files.
- Note: This script is supposed to be run once and generates
    two split files train.txt and test.txt.
## Usage examples:
1. `python data_splitter.py -d [path/to/data] --train 1 --test 1 -r 0`
    - `-d`: Recursively search all npy files in [path/to/data].
    - `-r`: Shuffle the files with rng with random state r=0.
    - `--train` and `--test`: Split the data files into
        train and test with ratio=1:1.
    Since the split path (`-s`) is not specified,
    save the list of train data files to [path/to/data]/train.txt and
    that for the test data to [path/to/data]/test.txt.
2. `python data_splitter.py -d [path/to/data] -s [split/file/path]`
    - Since `--train` and `--test` are not specified,
        use the default train-test ratio = 8:2.
    - Since `-r` is not given, the filenames are not shuffled.
    - `-s`: Save the list of train data files to [split/file/path]/train.txt
        and that for the test data to [split/file/path]/text.txt
3. `python data_splitter.py -d [path/to/data] --train 3 --valid 2 --test 1`
    - `--valid`: Use train valid test ratio = 3:2:1.


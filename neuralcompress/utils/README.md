# New data extractor and splitter

## Extractor:
1. **filename:** `data_extractor_crop.py`.
1. **Input:**
    - `input_dir`, [`-i`]:  folder containing the h5 files. (Currently we just need to input `/data/sphenix/sPHENIX_data/` as `input_dir`. Don't include the layer group in the `input_dir`).
    - `otuput_dir`, [`-o`]: the folder containing subfolders of npy files. (We just need to provide folder base and the subfolders will be generated automatically).
    - `layer_groups`, [`-g`] (optional): a string or a list of strings in `["inner", "middle", "outer"]`; If None, extract all;
    - `num_sectors`, [`-m`] (optional): number of sectors in the circle; 
        _The azimuthal dimension should be divisible by num_sectors_. Default value is $1$. 
    - `sectors`, [`-r`] (optional): an integer or a list of integers in `range(num_sectors)`; The sectors to keep; If None, extract all.
    - `num_sections`, [`-m`] (optional): the number of sections in the $z$ (horizontal) direction; _The $z$ dimension should be divisible by num_sections_. Default value is also $1$.
    - `sections`, [`-s`] (optional): an integer or a list of integer in `range(num_sections)`; The sections to keep; If None, extract all.
1. **output:** None.
1. **output file structure:** `[output_dir]/[layer_group]/[num_sectors]-[num_sections]_[sector]-[section]/[file_name]_[sample_id]_[layer].npy`
    > **Example:** we extract image from the `inner` layers,  cut the azimuthal dimension into $12$ sectors and the $z$ dimension into $2$ sections. The path to one of the file extracted looks like:  `./data/hight_framedata/inner/12-2_3-1/AuAu200_170kHz_10C_Iter2_3017.xml_TPCMLDataInterface_1_15.npy`.
1. **Examples:** In the folder containing `data_extractor_crop.py`
    - Extract the whole frame without cropping and do it for all three layers. 
        > `./data_extractor_crop.py -i /data/sphenix/sPHENIX_data/hightest_tpc/ -o [output_dir]` 
    - Cut the azimuthal dimension into $12$ sectors and extract all the sectors, cut the $z$ dimension into two halves and use the second half. 
        > `./data_extractor_crop.py -i /data/sphenix/sPHENIX_data/hightest_tpc/ -o [output_dir] -g outer -m 12 -n 2 -s 1`

## Splitter:
1. **filename:** `data_splitter_crop.py`
1. **input:**
    - `input_dir`, [`-i`]: input dir that contains npy files or subfolders (working with `input_dir_glob`) that contains npy files.
    - `output_dir`, [`-o`]: output dir that contains the splits (`train.txt`, `valid.txt`, and `text.txt`) or subfolders (working with `input_dir_glob`) that contain the splits.
    - `input_dir_glob` [`-g`] (optional): the pattern of subfolders (under `input_dir`) containing npy files; For example, when `input_dir_glob = "inner/12-2_*/"`, the function will look for all paths with the form `"[input_dir]/inner/12-2_*/"` for npy files. Suppose `"[input_dir]/inner/12-2_4-0"` is such a subfolder, then the corresponding split files are saved to `"[output_dir]/inner/12-2_4-0"`.
    - `max_examples` [`-m`] (optional): an integer that is the upper bound of the maximum number of total examples (train + valid + test) to use. If not provided, use all. 
    - ratios [`-s`] (optional): $3$ integers indicating the $\textrm{train} : \textrm{valid} : \textrm{test ratios}$. For example, suppose we have $10000$ npy files, and we use `ratios = (8, 1, 1)`, then we will have $8000$ training examples, $1000$ validation examples, and $1000$ test examples. 
    - `rnd_seed` [`-r`] (optional): an integer used for `np.random.seed(rnd_seed)`, default is None.
2. **output:** None
3. **output file structure:** 
    - *Case 1:* the npy files are contained directly in `input_dir` (`input_dir_glob=None`), then the output files will be `[output_dir]/train.txt`, `[output_dir]/valid.txt`, `[output_dir]/text.txt`. Each txt file contains symlinks (path) to the npy files in the corresponding split.
    - *Case 2:* the npy are contained in subfolders of `input_dir`. For simplicity, let us assume that the subfolders are `[input_dir]/inner/12-2_2-0/` and `[input_dir]/inner/12-2_2-1` obtained by using `input_dir_glob = "inner/12-2_2-*"` for example. Then the output files are 
        - `[output_dir]/inner/12-2_2-0/train.txt`
        - `[output_dir]/inner/12-2_2-0/valid.txt`
        - `[output_dir]/inner/12-2_2-0/test.txt`
        - `[output_dir]/inner/12-2_2-1/train.txt`
        - `[output_dir]/inner/12-2_2-1/valid.txt`
        - `[output_dir]/inner/12-2_2-1/test.txt`
4. **Example:** In the folder containing `data_splitter_crop.py`
    - Find npy files in `/data/sphenix/sPHENIX_data/outer`, and save the split files to `[output_file]/outer`. The splits are obtained with random seed $0$. $50\%$ of examples are used for training, $30\%$ for validation, and the remaining $20\%$ for test. 
        > `./data_splitter_crop.py -i /data/sphenix/sPHENIX_data/highest_framedata -o [output_dir] -g outer -r 0 -s 5 3 2`
    - Find npy files in `/data/sphenix/sPHENIX_data/outer`, and save the split files to `[output_file]`. The splits are obtained with random seed `None` and the default split ratios.
        > `./data_splitter_crop.py -i /data/sphenix/sPHENIX_data/highest_framedata/outer -o [output_dir]`
    - Find npy files in `/data/sphenix/sPHENIX_data/inner`, and save the split files to `[output_file]`. The splits are obtained with random seed $1$ and the default split ratios. Get $2000$ examples ($1600$ train, $200$ valid, $200$ test)
        > `./data_splitter_crop.py -i /data/sphenix/sPHENIX_data/highest_framedata/inner -o [output_dir] -r 1 -m 2000`
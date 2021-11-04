#!/usr/bin/env python

import h5py
import numpy as np
from pathlib import Path
from itertools import product
import argparse

# ============================ Dimensions ============================ START
layer_choice_map = {
    # The starting index of a group of layers
    'inner': 0,
    'middle': 16,
    'outer': 32
}
layer_dim_map = {
    # The azimuthal dimension of a group of layers
    'inner': 1152,
    'middle': 1536,
    'outer': 2304
}
layer_dim = 16
z_dim = 498
sample_dim = 10
# ============================ Dimensions ============================ END

def extract(
    input_dir, 
    output_dir, 
    layer_groups=None, 
    num_sectors=1, 
    sectors=None, 
    num_sections=1,
    sections=None):
    """ 
    **Data Extraction**

    This function extracts data from h5 files:
    See `EICLDRD/docs/data_description.md` for description of the h5 file.
    - The $z$ direction has dimension $498$.
    - Inner layer has dimension $1152$, with file size $\approx88$MB.
    - Middle layer has dimension $1536$, with file size $\approx117$MB.
    - Outer layer has dimension $2304$, withfile size $\approx210$MB.
    
    **note**: 
    - 498 = 83 * 2 * 3;
    - 1152 = (2^7) * (3^2); 
    - 1536 = (2^9) * 3; 
    - 2304 = (2^8) * (3^2).
    For now, it is enforced that we cut the circle and z direction 
    into a number of parts that divides the corresponding dimension.

    **input**:
    - input_dir: folder containing the h5 files.
    - otuput_dir: the folder containing npy files contains the each crop from each (sample, layer).
    - layer_groups: value or a list of values in ["inner", "middle", or "outer"]; 
        If None, extract all;
    - num_sectors: number of sectors in the circle; 
        The azimuthal dimension should be divisible by num_sectors. 
    - sectors: a list of numbers in {0, 1, ..., num_sectors - 1}; 
        The sectors to keep; 
        If None, keep all.
    - num_sections: number of sections in the "z" (horizontal) direction; 
        The z dimension should be divisible by num_sections.
    - sections: a list of numbers in {0, 1, ..., num_sections - 1}; 
        The sections to keep; 
        If None, keep all.
    
    **output file structure**:
    - [output_dir]/[layer_group]/[num_sectors]-[num_sections]_[sector]-[section]/[file_name]_[sample_id]_[layer].npy
    """

    # =========================== Parameters validity check ============================ START
    if layer_groups is None:
        layer_groups = ['inner', 'middle', 'outer']
    else:
        if not isinstance(layer_groups, str):
            layer_groups = [lg for lg in layer_groups if lg in layer_choice_map]
        else:
            if layer_groups in layer_choice_map:
                layer_groups = [layer_groups]
        
    for lg in layer_groups:
        assert layer_dim_map[lg] % num_sectors == 0, \
            f'number of sectors {num_sectors} must divide the dimension {layer_dim_map[lg]} of the {lg} layer'
        assert z_dim % num_sections == 0, \
            f'number of sections {num_sections} must divide the z dimension {z_dim}'

    input_dir = Path(input_dir)
    assert input_dir.is_dir(), f'{input_dir} does not exist!'

    if sectors is None:
        sectors = range(num_sectors)
    else:
        if not isinstance(sectors, int):
            assert all([sr >= 0 and sr < num_sectors for sr in sectors]), \
                f"all values in {sectors} should be in [0, ..., {num_sectors} - 1]"
        else:
            sectors = [sectors]
    if sections is None:
        sections = range(num_sections)
    else:
        if not isinstance(sections, int):
            assert all([sn >= 0 and sn < num_sections for sn in sections]), \
                f"all values in {sections} should be in [0, ..., {num_sections} - 1]"
        else:
            sections = [sections]
    
    print(f'\n================= Double-check the parameters =================\n')
    print(f'\tlayer groups to extract = {layer_groups}')
    print(f'\tnumber of sectors = {num_sectors}')
    print(f'\tsectors to extract = {sectors}')
    print(f'\tnumber of sections = {num_sections}')
    print(f'\tsections to extract = {sections}')
    print(f'\n===============================================================\n')
    if input('Continue?[Y/n]') != 'Y':
        print('Terminating')
        exit(1)
    # =========================== Parameters validity check ============================ END
    
    # =============================== making directories =============================== START
    output_dir = Path(output_dir)
    for (lg, sr, sn) in product(layer_groups, sectors, sections):
        output_subdir = Path(output_dir/f'{lg}/{num_sectors}-{num_sections}_{sr}-{sn}/')
        output_subdir.mkdir(parents=True, exist_ok=True)
    # =============================== making directories =============================== END

    
    # ==================================== Extract ===================================== START
    for lg in layer_groups:
        layer_start_idx = layer_choice_map[lg]
        azimuthal_dim = layer_dim_map[lg]
        a_block_size = azimuthal_dim // num_sectors
        z_block_size = z_dim // num_sections
        
        prefix = output_dir/f'{lg}/{num_sectors}-{num_sections}'

        h5_file_list = sorted(list(input_dir.glob('*.h5')))
        num_h5_files = len(h5_file_list)
        for i, file_name in enumerate(h5_file_list):
            stem = file_name.stem
            print(f'[{lg}], {i + 1}/{num_h5_files}: {stem}')
            with h5py.File(str(file_name), 'r') as fp:
                buf = np.zeros((azimuthal_dim, z_dim), dtype='uint16')
                for s, l in product(range(sample_dim), range(layer_dim)):
                    suffix = f'{s}_{layer_start_idx + l}'
                    key = f'/TimeFrame_{s}/Data_Layer{layer_start_idx + l}'
                    buf = fp[key][()]
                    for sr, sn in product(sectors, sections):
                        a_start, a_end = a_block_size * sr, a_block_size * (sr + 1)
                        z_start, z_end = z_block_size * sn, z_block_size * (sn + 1)
                        datum = buf[a_start: a_end, z_start: z_end]
                        # print(f'\tbuf[{a_start}: {a_end}, {z_start}: {z_end}], shape={datum.shape}')
                        np.save(Path(f'{prefix}_{sr}-{sn}/{stem}_{suffix}'), datum)
    # ==================================== Extract ===================================== END



if __name__ == '__main__':
    
    # input_dir = '/data/sphenix/sPHENIX_data/highest_tpc'
    # output_dir = './data/highest_framedata/'


    parser = argparse.ArgumentParser(description='Extract Data from h5 to npy.')
    parser.add_argument('-i', '--input_dir', type=str, required=True, help='input dir of h5 files.')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='output dir of the npy files.')
    parser.add_argument('-g', '--layer_groups', nargs='+', default=None, help="a single layer group or a \
        list of layer groups with values in ['inner', 'middle', 'outer]. \
        If not given, extract all three groups.")
    parser.add_argument('-m', '--num_sectors', type=int, default=1,
        help='number of sectors in the azimuthal direction.')
    parser.add_argument('-n', '--num_sections', type=int, default=1,
        help='number of sections in the z direction.')
    parser.add_argument('-r', '--sectors', nargs='+', type=int, default=None,
        help='sectors in the azimuthal direction to extract; \
        An integer or a list of integers in range(num_sectors); \
        If not given, extract all sectors.')
    parser.add_argument('-s', '--sections', nargs='+', type=int, default=None,
        help='sections in the z direction to extract; \
        An integer or a list of integers in range(num_sections); \
        If not given, extract all sections.')

    args = parser.parse_args()
    input_dir, output_dir = args.input_dir, args.output_dir
    layer_groups = args.layer_groups
    num_sectors, num_sections = args.num_sectors, args.num_sections
    sectors, sections = args.sectors, args.sections

    extract(
        input_dir, 
        output_dir, 
        layer_groups=layer_groups, 
        num_sectors=num_sectors,
        sectors=sectors,
        num_sections=num_sections,
        sections=sections
    )


    # for layer_group in ['inner', 'middle', 'outer']:
    # input_dir = f'/data/sphenix/sPHENIX_data/highest_framedata/{layer_group}'
    # output_dir = f'./data/highest_split/{layer_group}'
    # split(input_dir, output_dir, ratios=(8, 1, 1))

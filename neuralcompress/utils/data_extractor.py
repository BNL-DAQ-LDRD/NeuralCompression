"""
Extract data from specific group of layers
    - we cut the z dimension in half. 498 down to 249. see `EICLDRD/docs/data_description.md` for detail.
    - inner layer has smaller dimension, 1152. The filesize is about 88 MB
    - middle layer has dimension of 1536. The filesize is about 117 MB
    - outer layer has dimension of 2304. The filesize is about 210 MB
output:
    - `file_name_<time>_<layer>.npy`
"""

import argparse
import h5py
import numpy as np
from pathlib import Path

layer_choice_map = {"inner": 0, "middle": 16, "outer": 32}
layer_dim_map = {"inner": 1152, "middle": 1536, "outer": 2304}

def main(input_dir, output_dir, mode):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    assert input_dir.is_dir() and input_dir.exists()
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_sid, azimuthal_dim = layer_choice_map[mode], layer_dim_map[mode]
    # z_dim only uses first half
    sample_dim, layer_dim, z_dim = 10, 16, 249

    for file_name in input_dir.glob('*.h5'):
        print("processing filename", file_name)
        # ADC data is 10-bit int
        buf = np.zeros((azimuthal_dim, z_dim), dtype='uint16')
        with h5py.File(str(file_name), 'r') as fp:
            for t in range(sample_dim):
                for l in range(layer_dim):
                    key_name = f"/TimeFrame_{t}/Data_Layer{layer_sid+l}"
                    outfile = output_dir/f"{file_name.stem}_{t}_{layer_sid+l}"
                    # get first 249 in z dim
                    buf = (fp[key_name][()])[:, :z_dim]
                    np.save(outfile, buf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract Data from h5 to npy.')
    parser.add_argument('layer_group', type=str,
                        choices=list(layer_choice_map.keys()),
                        help='pick which layer group to extract')
    parser.add_argument('-i', '--input_dir', type=str,
                        help='input dir that contains h5 files')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='output dir that stores npy files')
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.layer_group)

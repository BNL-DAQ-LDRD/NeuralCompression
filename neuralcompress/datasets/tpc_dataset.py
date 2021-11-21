"""
TPC 3d Dataset
"""

#! /usr/bin/env python

from pathlib import Path
import numpy as np

from torch.utils.data import Dataset


class DatasetTPC3d(Dataset):
    """
    TPC 3d Dataset
    """
    def __init__(self, fname):
        """
        Input:
            - fname (str): The filename of the data manifest.
                Each line in file is an absolute path of a data file.
        """
        super().__init__()

        # validity check
        self.fname = Path(fname)
        assert self.fname.exists(), \
            f"The input split file, {fname}, does not exists!"

        # load split filenames
        with open(fname, 'r') as file_handle:
            self.file_list = file_handle.read().splitlines()

        self._verify_files()

    def _verify_files(self):
        for fname in self.file_list:
            assert Path(fname).exists(), \
                f'data file {fname} does not exists!'


    def __len__(self):
        return len(self.file_list)


    def __getitem__(self, idx):
        fname = self.file_list[idx]
        datum = np.expand_dims(np.float32(np.load(fname)), 0)
        return datum

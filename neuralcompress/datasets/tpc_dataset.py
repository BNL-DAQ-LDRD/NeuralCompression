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
    def __init__(self,split_fname):
        super().__init__()

        # validity check
        self.split_fname = Path(split_fname)
        assert self.split_fname.exists(), \
            f"The input split file, {split_fname}, does not exists!"

        # load split filenames
        with open(split_fname, 'r') as file_handle:
            self.file_list  = file_handle.read().splitlines()


    def __len__(self):
        return len(self.file_list)


    def __getitem__(self, idx):
        fname = self.file_list[idx]
        datum = np.expand_dims(np.float32(np.load(fname)), 0)
        return datum

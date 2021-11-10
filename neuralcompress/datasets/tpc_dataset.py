"""
TPC 3d Dataset
"""

#! /usr/bin/env python

from pathlib import Path
import numpy as np

from torch.utils.data import Dataset
# from torch.utils.data import DataLoader


class DatasetTPC3d(Dataset):
    """
    TPC 3d Dataset
    """
    #pylint: disable=too-many-arguments
    def __init__(
        self,
        split_path,
        frame_path,
        split='train',
    ):
        super().__init__()

        # validity check
        self.split_path = Path(split_path)
        assert self.split_path.exists(), \
            f"The input split path, {split_path}, does not exists!"

        split_fname = self.split_path/f'{split}.txt'
        assert split_fname.exists(), \
            f"Split file, {split_fname}, does not exists!"

        self.frame_path = Path(frame_path)
        assert self.frame_path.exists(), \
            f"The input frame path, {frame_path}, does not exists!"

        # load split filenames
        with open(split_fname, 'r') as file_handle:
            file_names = file_handle.read().splitlines()
            self.file_list = [frame_path/f for f in file_names]


    def __len__(self):
        return len(self.file_list)


    def __getitem__(self, idx):
        fname = self.file_list[idx]
        datum = np.expand_dims(np.float32(np.load(fname)), 0)
        return datum


# #pylint: disable=too-many-arguments
# def main(
#     data_root,
#     layer_group,
#     batch_size,
# ):
#     """
#     This is main of dataset_tpc.py.
#     It serves as the test for the DatasetTPC3d API.
#     """
#
#     # dataset paths
#     split_path = Path(data_root)/f'highest_split_3d/{layer_group}/'
#     frame_path = Path(data_root)/f'highest_framedata_3d/{layer_group}/'
#
#     # datasets
#     dataset_train = DatasetTPC3d(
#         split_path,
#         frame_path,
#         split='train',
#     )
#     dataset_valid = DatasetTPC3d(
#         split_path,
#         frame_path,
#         split='valid',
#     )
#     dataset_test  = DatasetTPC3d(
#         split_path,
#         frame_path,
#         split='test',
#     )
#
#     # dataloaders
#     dl_train = DataLoader(dataset_train, batch_size=batch_size)
#     dl_valid = DataLoader(dataset_valid, batch_size=batch_size)
#     dl_test  = DataLoader(dataset_test,  batch_size=batch_size)
#
#     for data_loader in [dl_train, dl_valid, dl_test]:
#         for i, batch in enumerate(data_loader):
#             print(f'{i + 1}/{len(data_loader)}: {len(batch)}')
#
# if __name__ == '__main__':
#     main(
#         '/data/datasets/sphenix/',
#         'outer',
#         32,
#     )

#! /usr/bin/env python
import sys
from torch.utils.data import DataLoader

sys.path.append('/home/yhuang2/PROJs/NeuralCompression/neuralcompress/')

from datasets.tpc_dataset import DatasetTPC3d
from utils.data_splitter import Splitter


def get_dataloaders(
    data_path,
    shuffle=True,
    rng_random_state=0,
    max_sizes=None,
    batch_size=32
):
    split_path = data_path

    splitter = Splitter(
        data_path,
        split_path,
        max_sizes=max_sizes,
        shuffle=shuffle,
        rng_random_state=rng_random_state
    )

    splitter.split()

    dataset_train = DatasetTPC3d(split_path, data_path, split='train')
    dataset_valid = DatasetTPC3d(split_path, data_path, split='valid')
    dataset_test  = DatasetTPC3d(split_path, data_path, split='test' )

    loader_train = DataLoader(dataset_train, batch_size=batch_size)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size)
    loader_test  = DataLoader(dataset_test,  batch_size=batch_size)

    loaders = [loader_train, loader_valid, loader_test]
    return loaders


if __name__ == "__main__":
    print('This is main of data_loader.py')

    # frame_path = '/data/datasets/sphenix/highest_framedata_3d/outer/'
    # max_sizes = [300, 100, 100]
    # shuffle = True
    # rng_random_state = 0
    # batch_size = 32

    # loaders = get_dataloaders(
    #     frame_path,
    #     shuffle=shuffle,
    #     rng_random_state=rng_random_state,
    #     max_sizes=max_sizes,
    #     batch_size=batch_size
    # )

    # for loader in loaders:
    #     for i, batch in enumerate(loader):
    #         print(f'{i + 1}/{len(loader)}: {len(batch)}')
    #     print()

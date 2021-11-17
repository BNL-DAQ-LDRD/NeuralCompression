# def test():
#     """
#     Test with TPC dataset API.
#     """
#     from pathlib import Path
#     from neuralcompress.datasets.tpc_dataset import DatasetTPC3d
#
#     batch_size = 32
#     root_folder = Path('/data/datasets/sphenix/highest_framedata_3d/outer/')
#     train_loader, valid_loader = get_dataloader(
#         DatasetTPC3d,
#         root_folder/'train.txt',
#         batch_size,
#         lengths=[300, 100],
#         seed=0
#     )
#
#     print('\ntrain')
#     for i, batch in enumerate(train_loader):
#         print(f'{i + 1}/{len(train_loader)}: {len(batch)}')
#     print('\nvalid')
#     for i, batch in enumerate(valid_loader):
#         print(f'{i + 1}/{len(valid_loader)}: {len(batch)}')
#
#     loader = get_dataloader(
#         DatasetTPC3d,
#         root_folder/'test.txt',
#         batch_size=32,
#         lengths=200
#     )
#     print('\ntest')
#     for i, batch in enumerate(loader):
#         print(f'{i + 1}/{len(loader)}: {len(batch)}')


if __name__ == "__main__":
    print('This is main of data_loader.py')
    test()

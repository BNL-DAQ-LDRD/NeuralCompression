"""
# Test the TPC dataloader function

# Usage example:
    1. python -m test_tpc_dataloader -p [manifest_path] -b 32 --train_sz 32\
         --valid_sz 32 --test_sz 32  --shuffle --seed 0
        > Load 32 train examples, 32 valid examples, 32 test examples.
        Shuffle the dataset and seed the random number generator with 0.
    2. python -m test_tpc_dataloader -p [manifest_path] -b 32 --valid_ratio 1\
         --no-shuffle
        > Load all examples in the train manifest and split it into equal\
        number of train and valid examples. Load all examples in the test
        manifest. Do not shuffle the dataset.
"""
import argparse
from neuralcompress.utils.tpc_dataloader import get_tpc_dataloaders

#pylint: disable=too-many-arguments
def test(
    manifest_path,
    batch_size,
    train_sz    =None,
    valid_sz    =None,
    valid_ratio =None,
    test_sz     =None,
    shuffle     =True,
    seed        =None,
):
    """
    Test the TPC dataloader function
    """
    train_loader, valid_loader, test_loader = get_tpc_dataloaders(
        manifest_path,
        batch_size,
        train_sz    = train_sz,
        valid_sz    = valid_sz,
        valid_ratio = valid_ratio,
        test_sz     = test_sz,
        shuffle     = shuffle,
        seed        = seed,
    )

    print('\ntrain')
    for i, batch in enumerate(train_loader):
        print(f'{i + 1}/{len(train_loader)}: {len(batch)}')

    print('\nvalid')
    for i, batch in enumerate(valid_loader):
        print(f'{i + 1}/{len(valid_loader)}: {len(batch)}')

    print('\ntest')
    for i, batch in enumerate(test_loader):
        print(f'{i + 1}/{len(test_loader)}: {len(batch)}')


def get_arguments():
    """
    Get command line argument
    """
    parser = argparse.ArgumentParser(
        description='Test TPC dataloader function.'
    )
    parser.add_argument(
        '--manifest_path',
        '-p',
        type=str,
        required=True,
        help='The path to the manifest file. \
            Must contain train.txt, valid.txt, and test.txt.'
    )
    parser.add_argument(
        '--batch_size',
        '-b',
        type=int,
        required=True,
        help='batch size.'
    )
    parser.add_argument(
        '--train_sz',
        type=int,
        required=False,
        default=None,
        help='train size.'
    )
    parser.add_argument(
        '--valid_sz',
        type=int,
        required=False,
        default=None,
        help='valid size.'
    )
    parser.add_argument(
        '--valid_ratio',
        type=float,
        required=False,
        default=None,
        help='valid-to-test ratio \
        Setting valid_ratio=1 means we have \
        the same number of train and valid examples.'
    )
    parser.add_argument(
        '--test_sz',
        type=int,
        required=False,
        default=None,
        help='test size.'
    )
    parser.add_argument(
        '--shuffle',
        dest='shuffle',
        action='store_true',
        help='shuffle the dataset.'
    )
    parser.add_argument(
        '--no-shuffle',
        dest='shuffle',
        action='store_false',
        help='do not shuffle the dataset.'
    )
    parser.set_defaults(shuffle=True)
    parser.add_argument(
        '--seed',
        type=int,
        required=False,
        default=None,
        help='random seed.'
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    test(
        manifest_path = args.manifest_path,
        batch_size    = args.batch_size,
        train_sz      = args.train_sz,
        valid_sz      = args.valid_sz,
        valid_ratio   = args.valid_ratio,
        test_sz       = args.test_sz,
        shuffle       = args.shuffle,
        seed          = args.seed
    )

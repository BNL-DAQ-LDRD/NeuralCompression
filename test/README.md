# Test the TPC dataloader function
Use `test_tpc_dataloader.py` to test the working of TPC dataloader
## Usage examples:
1. `python -m test_tpc_dataloader -p [manifest_path] -b 32 --train_sz 32 --valid_sz 32 --test_sz 32  --shuffle --seed 0`
    > Load 32 train examples, 32 valid examples, 32 test examples.
    Shuffle the dataset and seed the random number generator with 0.
1. `python -m test_tpc_dataloader -p [manifest_path] -b 32 --valid_ratio 1 --no-shuffle`
    > Load all examples in the train manifest and split it into equal number of train and valid examples.
    Load all examples in the test manifest. Do not shuffle the dataset.

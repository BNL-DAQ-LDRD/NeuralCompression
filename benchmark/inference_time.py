"""
Inference time study
"""
import argparse
from pathlib import Path
from time import time
import pandas as pd
import numpy as np

import torch
import torch.backends

from neuralcompress.utils.load_bcae_models import load_bcae_encoder
from neuralcompress.utils.tpc_dataloader import get_tpc_dataloaders


DATA_ROOT = '/data/datasets/sphenix/highest_framedata_3d/outer'
CHECKPOINTS = '/home/yhuang2/PROJs/NeuralCompression_results/checkpoints/'
EPOCH = 440

def inference():
    """
    inference time study
    """

    parser = argparse.ArgumentParser(
        description="Run BCAE inference"
    )

    parser.add_argument(
        '--data_size',
        required = False,
        default  = 1,
        type     = int,
        help     = "Number of frames to load. (default=1)"
    )
    parser.add_argument(
        '--batch_size',
        required = False,
        default  = 1,
        type     = int,
        help     = "Batch size. (default=1)"
    )
    parser.add_argument(
        '--num_runs',
        required = False,
        default  = 10,
        type     = int,
        help     = "Number of runs to calculate the run-time. (default=10)"
    )
    parser.add_argument(
        # If True, the data loader will copy Tensors
        # into CUDA pinned memory before returning them.
        '--pin_memory',
        action   = 'store_true',
        help     = "If True, the dataloader will copy Tensors into CUDA \
                    pinned memory before returning them."
    )
    parser.add_argument(
        '--num_workers',
        required = False,
        default  = 0,
        type     = int,
        help     = "how many subprocesses to use for data loading. \
                    0 means that the data will be loaded in the main process. \
                    (default=0)"
    )
    parser.add_argument(
        '--prefetch_factor',
        required = False,
        default  = 2,
        type     = int,
        help     = "Number of samples loaded in advance by each worker. \
                    (default=2)"
    )
    parser.add_argument(
        '--benchmark',
        action   = 'store_true',
        help     = "If True, set torch.backends.cudnn.benchmark to True."
    )

    parser.add_argument(
        '--result_fname',
        required = False,
        default  = 'result.csv',
        type     = str,
        help     = "Result filename. (default=result.csv)"
    )

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = False

    # Load data
    data_config = {
        'batch_size'      : args.batch_size,
        'train_sz'        : args.data_size,
        'valid_sz'        : 0,
        'test_sz'         : 0,
        'is_random'       : True,
        'pin_memory'      : args.pin_memory,
        'num_workers'     : args.num_workers,
        'prefetch_factor' : args.prefetch_factor
    }

    data_path = Path(DATA_ROOT)
    loader, _, _ = get_tpc_dataloaders(data_path, **data_config)

    # Load encoder
    encoder = load_bcae_encoder(CHECKPOINTS, EPOCH)
    encoder.to('cuda')

    res = vars(args).copy()
    num_runs = args.num_runs
    mb = 1024 ** 2
    records = []
    with torch.no_grad():
        # Run one epoch before timing so that cuda can adjust well.
        for batch in loader:
            _ = encoder(batch.to('cuda'))

        res['memory allocated (MB)'] = torch.cuda.memory_allocated()/mb
        res['memory reserved  (MB)'] = torch.cuda.memory_reserved()/mb
        res['max memory allocated (MB)'] = torch.cuda.max_memory_allocated()/mb
        res['max memory reserved  (MB)'] = torch.cuda.max_memory_reserved()/mb

        time0 = time()
        for _ in range(num_runs):
            time_sub = time()
            for batch in loader:
                _ = encoder(batch.to('cuda'))
            torch.cuda.synchronize()
            records.append(time() - time_sub)
        torch.cuda.synchronize()
        time1 = time()

    multiplier = 100 / (args.data_size * num_runs)
    res['inference time per hundred frames'] = (time1 - time0) * multiplier
    # res['mean'] = np.mean(records) * multiplier * num_runs
    res['std'] = np.std(records) * multiplier * num_runs

    del res['result_fname']
    df = pd.DataFrame(data=res, index=[1]).T
    df.to_csv(args.result_fname)
    print(df)

    return res

if __name__ == '__main__':
    inference()

"""
Inference time study
"""
import argparse
from time import time
import pandas as pd
import numpy as np

import torch
import torch.backends

from neuralcompress.models.bcae_encoder import BCAEEncoder
from neuralcompress.utils.tpc_dataloader import get_tpc_dataloaders

# when running on wavelet, please use the following data root
# DATA_ROOT = '/data/datasets/sphenix/highest_framedata_3d/outer'

def parse_args():
    """
    inference time study
    """
    parser = argparse.ArgumentParser(
        description="Run BCAE inference"
    )

    # Positional argument
    parser.add_argument(
        'checkpoint',
        type=str,
        help="The path to the encoder pt file."
    )

    # Optional
    parser.add_argument(
        '--num_runs',
        required = False,
        default  = 10,
        type     = int,
        help     = "Number of runs to calculate the run-time. (default=10)"
    )

    parser.add_argument(
        '--benchmark',
        action   = 'store_true',
        help     = "If used, set torch.backends.cudnn.benchmark to True."
    )

    parser.add_argument(
        '--with_loader',
        action   = 'store_true',
        help     = "If used, use TPC dataloader. \
                   Use randomly generated data if otherwise"
    )

    parser.add_argument(
        '--data_root',
        required = False,
        default  = None,
        type     = str,
        help     = "Path to data, when loading data with TPC dataloader \
                    (--with_loader). (default=None)"
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
        # If True, the data loader will copy Tensors
        # into CUDA pinned memory before returning them.
        '--pin_memory',
        action   = 'store_true',
        help     = "If used, the dataloader will copy Tensors into CUDA \
                    pinned memory before returning them."
    )

    parser.add_argument(
        '--num_workers',
        required = False,
        default  = 0,
        type     = int,
        help     = "Number of subprocesses to use for data loading. \
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
        '--result_fname',
        required = False,
        default  = 'result.csv',
        type     = str,
        help     = "Result filename. (default=result.csv)"
    )

    parser.add_argument(
        '--half_precision',
        action   = 'store_true',
        help     = "If used, run inference with half precision (float16)"
    )

    args = parser.parse_args()
    return args


def collect_gpu_memory():
    res = {}
    megabytes = 1024 ** 2
    res['memory allocated (MiB)']     = torch.cuda.memory_allocated()/megabytes
    res['memory reserved  (MiB)']     = torch.cuda.memory_reserved()/megabytes
    res['max memory allocated (MiB)'] = torch.cuda.max_memory_allocated()/megabytes
    res['max memory reserved  (MiB)'] = torch.cuda.max_memory_reserved()/megabytes
    return res

def inference():
    args = parse_args()

    torch.backends.cudnn.benchmark = args.benchmark

    # Load data
    assert not args.with_loader or (args.with_loader and args.data_root), \
        'If with_loader, data_root must be provided.'

    data_size  = args.data_size
    batch_size = args.batch_size
    if args.with_loader:
        data_config = {
            'batch_size'      : batch_size,
            'train_sz'        : data_size,
            'valid_sz'        : 0,
            'test_sz'         : 0,
            'is_random'       : False,
            'pin_memory'      : args.pin_memory,
            'num_workers'     : args.num_workers,
            'prefetch_factor' : args.prefetch_factor
        }
        data, _, _ = get_tpc_dataloaders(args.data_root, **data_config)
    else:
        assert data_size % batch_size == 0, \
            'data size must be a multiple of batch size'
        num_batches = data_size // batch_size
        data = torch.rand(num_batches, batch_size, 1, 192, 249, 16)
        data = data.to('cuda')


    # Load encoder
    encoder = BCAEEncoder()
    encoder.load_state_dict(torch.load(args.checkpoint))
    encoder.to('cuda')

    # Run inference and save results
    res = vars(args).copy()
    num_runs = args.num_runs
    records = []

    if args.half_precision:
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                if args.with_loader:
                    for batch in data:
                        batch = batch.to('cuda')
                        _ = encoder(batch)
                else:
                    for batch in data:
                        _ = encoder(batch)

                res.update(collect_gpu_memory())

                time0 = time()
                for _ in range(num_runs):
                    time_sub = time()
                    if args.with_loader:
                        for batch in data:
                            batch = batch.to('cuda')
                            _ = encoder(batch)
                    else:
                        for batch in data:
                            _ = encoder(batch)
                    torch.cuda.synchronize()
                    records.append(time() - time_sub)
                torch.cuda.synchronize()
                time1 = time()
    else:
        with torch.no_grad():
            if args.with_loader:
                for batch in data:
                    batch = batch.to('cuda')
                    _ = encoder(batch)
            else:
                for batch in data:
                    _ = encoder(batch)

            res.update(collect_gpu_memory())

            time0 = time()
            for _ in range(num_runs):
                time_sub = time()
                if args.with_loader:
                    for batch in data:
                        batch = batch.to('cuda')
                        _ = encoder(batch)
                else:
                    for batch in data:
                        _ = encoder(batch)
                torch.cuda.synchronize()
                records.append(time() - time_sub)
            torch.cuda.synchronize()
            time1 = time()

    res['frames per second'] = data_size * num_runs / (time1 - time0)
    res['std'] = np.std([data_size / r  for r in records])

    del res['result_fname']
    result_df = pd.DataFrame(data=res, index=[1]).T
    result_df.to_csv(args.result_fname)
    print(result_df)

    return res

if __name__ == '__main__':
    inference()

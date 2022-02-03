"""
Inference time study
"""
import argparse
# from pathlib import Path
from time import time
import pandas as pd
import numpy as np

import torch
import torch.backends

from neuralcompress.utils.load_bcae_models import load_bcae_encoder
# from neuralcompress.utils.tpc_datadummy_data import get_tpc_datadummy_datas


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

    # Get dummy input tensor
    data_size = args.data_size
    batch_size = args.batch_size
    assert data_size % batch_size == 0, \
        'data size must be a multiple of batch size'
    num_batches = data_size // batch_size
    dummy_data = torch.rand(num_batches, batch_size, 1, 192, 249, 16)
    dummy_data = dummy_data.to('cuda')

    # Load encoder
    encoder = load_bcae_encoder(CHECKPOINTS, EPOCH)
    encoder.to('cuda')

    res = vars(args).copy()
    num_runs = args.num_runs
    megabytes = 1024 ** 2
    records = []
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            # Run one epoch before timing so that cuda can adjust well.
            for batch in dummy_data:
                _ = encoder(batch)

            res['memory allocated (MB)'] = torch.cuda.memory_allocated()/megabytes
            res['memory reserved  (MB)'] = torch.cuda.memory_reserved()/megabytes
            res['max memory allocated (MB)'] = torch.cuda.max_memory_allocated()/megabytes
            res['max memory reserved  (MB)'] = torch.cuda.max_memory_reserved()/megabytes

            time0 = time()
            for _ in range(num_runs):
                time_sub = time()
                for batch in dummy_data:
                    output = encoder(batch)
                    print(output.shape)
                torch.cuda.synchronize()
                records.append(time() - time_sub)
            torch.cuda.synchronize()
            time1 = time()

    multiplier = (data_size * num_runs)
    res['frames per second'] = multiplier / (time1 - time0)
    # res['mean'] = np.mean(records) * multiplier * num_runs
    # res['std'] = np.std(records) * multiplier * num_runs
    res['std'] = np.std([data_size / r  for r in records])

    del res['result_fname']
    result_df = pd.DataFrame(data=res, index=[1]).T
    result_df.to_csv(args.result_fname)
    print(result_df)

    return res

if __name__ == '__main__':
    inference()

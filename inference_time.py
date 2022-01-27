"""
Inference time study
"""
from pathlib import Path
import tqdm
import numpy as np
from neuralcompress.utils.load_bcae_models import load_bcae_encoder
from neuralcompress.utils.tpc_dataloader import get_tpc_dataloaders
import torch
import torch.backends
cudnn.benchmark = True

# dataloader number of workers and pin_memory
# pre-fetching = 1 or 2


DATA_ROOT = '/data/datasets/sphenix/highest_framedata_3d/outer'

def inference():

    parser = argparse.ArgumentParser(
        description="Run BCAE inference"
    )


    parser.add_argument(
        '--device',
        required = False,
        default  = 'cuda',
        choices  = ['cuda', 'cpu'],
        type     = str,
        help     = "The device to run the inference."
    )

    parser.add_argument(
        '--data_size',
        required = False,
        default  = 1,
        type     = int,
        help     = "Number of frames to load | default=1."
    )

    parser.add_argument(
        '--batch_size',
        required = False,
        default  = 1,
        type     = int,
        help     = "Batch size | default=1."
    )

    parser.add_argument(
        '--partition',
        required = False,
        default  = 'test',
        choices=['train', 'valid', 'test'],
        type     = str,
        help     = "partition from which to load the data | default=test."
    )

    parser.add_argument(
        '--random',
        action = 'store_true',
        help   = "Whether to get a random sample."
    )

    parser.add_argument(
        '--checkpoint_path',
        required=True,
        type=str,
        help="The path to the checkpoints."
    )

    parser.add_argument(
        '--epoch',
        required=True,
        type=int,
        help="The epoch to load."
    )

    parser.add_argument(
        '--save_path',
        required = True,
        type     = str,
        help     = "The path to save output tensor."
    )

    parser.add_argument(
        '--half',
        action = 'store_true',
        help   = "Whether to save the output with half precision."
    )


    parser.add_argument(
        '--prefix',
        required = False,
        default  = 'output',
        type     = str,
        help     = "Output file prefix."
    )

    args = parser.parse_args()


    # Load data
    data_config = {
        'batch_size' : args.batch_size,
        'train_sz'   : 0,
        'valid_sz'   : 0,
        'test_sz'    : 0,
        'is_random'  : args.random,
    }
    data_config[f'{args.partition}_sz'] = args.data_size

    data_path = Path(args.data_path)
    assert data_path.exists(), f'{data_path} does not exist!'

    loaders = get_tpc_dataloaders(data_path, **data_config)
    partition = args.partition
    if partition == 'train':
        loader = loaders[0]
    elif partition == 'valid':
        loader = loaders[1]
    else:
        loader = loaders[2]


    # Load encoder
    checkpoint_path = Path(args.checkpoint_path)
    assert checkpoint_path.exists(), f'{checkpoint_path} does not exist!'

    encoder = load_bcae_encoder(checkpoint_path, args.epoch)
    encoder.to(args.device)

    # run inference
    # progbar = tqdm.tqdm(
    #     desc="BCAE Inference",
    #     total=len(loader),
    #     dynamic_ncols=True
    # )
    # outputs = []

    res = {}

    with torch.no_grad():
        for i, batch in enumerate(loader):
            output = encoder(batch.to(args.device))

        res['    memory allocated (MB)'.strip()] = torch.cuda.memory_allocated()/1024/1024
        res['    memory cached    (MB)'.strip()] = torch.cuda.memory_cached()/1024/1024
        res['max memory allocated (MB)'.strip()] = torch.cuda.max_memory_allocated()/1024/1024
        res['max memory cached    (MB)'.strip()] = torch.cuda.max_memory_cached()/1024/1024


        time0 = time()
        T = 10
        for _ in range(T):
            for i, batch in enumerate(loader):
                output = encoder(batch.to(args.device))
                # outputs.append(output.detach().cpu().numpy())
                # progbar.update()
            # progbar.close()
        torch.cuda.synchronize()
        time1 = time()

    # save_path = Path(args.save_path)
    # if not save_path.exists():
    #     save_path.mkdir(parents=True)

    # # save result
    # counter = 0
    # for output in outputs:
    #     for frame in output:
    #         if args.half:
    #             frame = frame.astype('float16')
    #         fname = save_path/f'{args.prefix}_{counter}'
    #         np.savez(fname, data=frame)
    #         counter += 1

if _# _name__ == '__main__':
    # inference()

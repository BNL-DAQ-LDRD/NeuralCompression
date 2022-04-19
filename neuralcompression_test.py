#!/usr/bin/env python

from pathlib import Path
import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn

from neuralcompress.utils.tpc_dataloader import get_tpc_dataloaders
from neuralcompress.utils.load_bcae_models import (
    load_bcae_encoder,
    load_bcae_decoder
)
from neuralcompress.models.bcae_combine import BCAECombine


# # Load a lot of test data
# # Use the following code if you are on the wavelet machine.
# data_path   = '/data/datasets/sphenix/highest_framedata_3d/outer/'
# data_config = {
#     'batch_size' : 64,
#     'train_sz'   : 0,
#     'valid_sz'   : 0,
#     'test_sz'    : 2048,
#     'is_random'  : True,
# }
# _, _, loader = get_tpc_dataloaders(data_path, **data_config)


# load the 8 frames in the repo
data_path   = 'data/'
data_config = {
    'batch_size' : 2,
    'train_sz'   : 0,
    'valid_sz'   : 0,
    'test_sz'    : 8,
    'is_random'  : True,
}
_, _, loader = get_tpc_dataloaders(data_path, **data_config)


device = 'cuda'
# Load encoder
checkpoint_path = Path('checkpoints')
epoch           = 2000
encoder = load_bcae_encoder(checkpoint_path, epoch)
decoder = load_bcae_decoder(checkpoint_path, epoch)
encoder.to(device)
decoder.to(device)

# run compression and decompression
combine = BCAECombine(threshold=.5)
loss_mse = nn.MSELoss()
loss_l1 = nn.L1Loss()

results = defaultdict(list)

with torch.no_grad():
    for batch in loader:
        batch  = batch.to(device)
        comp   = encoder(batch)
        decomp = combine(decoder(comp))

        mse  = loss_mse(decomp, batch).item()
        l1   = loss_l1(torch.log2(decomp + 1), torch.log2(batch + 1)).item()
        psnr = np.log10(1023 ** 2 / mse)

        results['mse'].append(mse)
        results['log_mae'].append(l1)
        results['psnr'].append(psnr)

        print(f'mse = {mse:.3f}, log mae = {l1: .3f}, psnr = {psnr: .3f}')

df = pd.DataFrame(data=results)
descr = df.describe().loc[['mean', 'std'], :]
print(descr)

from pathlib import Path
from collections import defaultdict
import tqdm
import numpy as np
import pandas as pd

import torch

from neuralcompress.utils.tpc_dataloader import get_tpc_dataloaders
from neuralcompress.utils.load_bcae_models import (
    load_bcae_encoder,
    load_bcae_decoder
)

from neuralcompress.models.bcae_combine import BCAECombine


#################################################################
# =================== Compress and decompress ===================
# Load data
data_path   = Path('/data/datasets/sphenix/highest_framedata_3d/outer/')
data_config = {
    'batch_size' : 32,
    'train_sz'   : 0,
    'valid_sz'   : 0,
    'test_sz'    : 320, # there are only 8 data files contained
    'is_random'  : False,
    'shuffle'    : False,
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
combine = BCAECombine()
progbar = tqdm.tqdm(
    desc="BCAE recall study",
    total=len(loader),
    dynamic_ncols=True
)
threshold = .5
recalls = []
with torch.no_grad():
    for i, batch in enumerate(loader):
        batch  = batch.to(device)

        comp   = encoder(batch)
        comp   = comp.half() # we save the compressed result as half float
        decomp = combine(decoder(comp.float()))

        positive = torch.count_nonzero(batch)
        true_pos = torch.count_nonzero((decomp > 0) * (batch > 0))

        recall = (true_pos / positive).item()
        recalls.append(recall)

        progbar.update()
    progbar.close()

mean = np.mean(recalls)
std  = np.std(recalls)

print(f'mean = {mean:.6f}, std = {std:.6f}')

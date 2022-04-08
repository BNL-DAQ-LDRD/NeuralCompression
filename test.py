from pathlib import Path
import tqdm
import numpy as np

import torch

from neuralcompress.utils.tpc_dataloader import get_tpc_dataloaders
from neuralcompress.utils.load_bcae_models import (
    load_bcae_encoder,
    load_bcae_decoder
)

from neuralcompress.models.bcae_combine import BCAECombine

# Load data
data_path   = 'data'
data_config = {
    'batch_size' : 4,
    'train_sz'   : 0,
    'valid_sz'   : 0,
    'test_sz'    : 8, # there are only 32 data files contained
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
combine = BCAECombine()
compressed   = []
decompressed = []
progbar = tqdm.tqdm(
    desc="BCAE compression and decompression",
    total=len(loader),
    dynamic_ncols=True
)

with torch.no_grad():
    for batch in loader:
        batch  = batch.to(device)

        comp   = encoder(batch)
        decomp = combine(decoder(comp))

        compressed.append(comp.detach().cpu().numpy())
        decompressed.append(decomp.detach().cpu().numpy())

        progbar.update()
    progbar.close()

# save result
save_path = Path('results')
counter = 0
for comp, decomp in zip(compressed, decompressed):
    for en, de in zip(comp, decomp):
        en = en.astype('float16')
        np.savez(save_path/f'compressed_{counter}', data=en)
        np.savez(save_path/f'decompressed_{counter}', data=de)
        counter += 1

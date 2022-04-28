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



# =================== Compress and decompress ===================

# Load data
data_path   = 'data'
data_config = {
    'batch_size' : 4,
    'train_sz'   : 0,
    'valid_sz'   : 0,
    'test_sz'    : 8, # there are only 8 data files contained
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
        comp   = comp.half() # we save the compressed result as half float
        decomp = combine(decoder(comp.float()))

        compressed.append(comp.detach().cpu().numpy())
        decompressed.append(decomp.detach().cpu().numpy())

        progbar.update()
    progbar.close()

# save result
save_path = Path('results')
counter = 0
for comp, decomp in zip(compressed, decompressed):
    for en, de in zip(comp, decomp):
        de = np.squeeze(de)
        np.savez(save_path/f'compressed_{counter}', data=en)
        np.savez(save_path/f'decompressed_{counter}', data=de)
        counter += 1



#=================== Load result and compute metrices ===================

print('\nCompute MSE, log-MAE, and PSNR')
# load input
fnames_raw = sorted(list(Path('./data').glob('*npy')))
# load reconstructed image
fnames_rec = sorted(list(Path('./results').glob('decomp*npz')))

# metrics
results = defaultdict(list)

for i, (fname_raw, fname_rec) in enumerate(zip(fnames_raw, fnames_rec)):
    raw = np.load(fname_raw)
    with np.load(fname_rec) as fh:
        rec = fh[fh.files[0]]

    # mean squared error and peak signal-noise ratio
    diff = np.abs(raw - rec)
    mse  = np.mean(diff * diff)
    psnr = np.log10(1023 ** 2 / mse)
    # Log Mean absolute error
    log_raw  = np.log2(raw + 1)
    log_rec  = np.log2(rec + 1)
    log_diff = np.abs(log_raw - log_rec)
    log_mae  = np.mean(log_diff)

    results['mse'].append(mse)
    results['log_mae'].append(log_mae)
    results['psnr'].append(psnr)

    print(f'\tsample {i}: mse = {mse:.3f}, log mae = {log_mae: .3f}, psnr = {psnr: .3f}')

df = pd.DataFrame(data=results)
descr = df.describe().loc[['mean', 'std'], :]
print()
print(descr)

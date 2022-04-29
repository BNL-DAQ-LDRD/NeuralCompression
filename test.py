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
data_path   = Path('./data')
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
originals    = []
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

        originals.append(batch.detach().cpu().numpy())
        compressed.append(comp.detach().cpu().numpy())
        decompressed.append(decomp.detach().cpu().numpy())

        progbar.update()
    progbar.close()

# reshape the tensors
originals    = np.squeeze(np.vstack(originals))
compressed   = np.vstack(compressed)
decompressed = np.squeeze(np.vstack(decompressed))

# save result
save_path = Path('results')
if not save_path.exists():
    save_path.mkdir()

for i, (comp, decomp) in enumerate(zip(compressed, decompressed)):
    np.save(save_path/f'compressed_{i}', comp)
    np.save(save_path/f'decompressed_{i}', decomp)



###################################################################
# =================== Sanity check with metrics ===================
print('\n============== Recontruction Errors ===============')
print('Metrics obtained from the current run:')
metrics = defaultdict(list)
for i, (orig, decomp) in enumerate(zip(originals, decompressed)):
    # mean squared error and peak signal-noise ratio
    diff = np.abs(orig - decomp)
    mse  = np.mean(diff * diff)
    psnr = np.log10(1023 ** 2 / mse)
    # Log Mean absolute error
    log_orig   = np.log2(orig + 1)
    log_decomp = np.log2(decomp + 1)
    log_diff   = np.abs(log_orig - log_decomp)
    log_mae    = np.mean(log_diff)

    metrics['mse'].append(mse)
    metrics['log_mae'].append(log_mae)
    metrics['psnr'].append(psnr)

df = pd.DataFrame(data=metrics)
descr = df.describe().loc[['mean', 'std'], :]
print(f'{df}\n{descr}')

print('\nExpected metrics:')
df_exp = pd.read_csv(data_path/'sample_metric_results.csv', index_col=0)
descr_exp = df_exp.describe().loc[['mean', 'std'], :]
print(f'{df_exp}\n{descr_exp}')


##########################################################################
# =================== Load result and compute metrices ===================

print('\n========= Compare current output with cached ones =========')
# load compressed and decompressed filenames
fnames_comp   = sorted(list(Path('./data').glob('comp*npy')))
fnames_decomp = sorted(list(Path('./data').glob('decomp*npy')))

for i, (comp, fname_comp) in enumerate(zip(compressed, fnames_comp)):
    comp_cached = np.load(fname_comp)
    diff = np.abs(comp - comp_cached)
    mse  = np.mean(diff * diff)
    print(f'Sample {i} compressed: MSE = {mse:.3e}')

for i, (decomp, fname_decomp) in enumerate(zip(decompressed, fnames_decomp)):
    decomp_cached = np.load(fname_decomp)
    diff = np.abs(decomp - decomp_cached)
    mse  = np.mean(diff * diff)
    print(f'Sample {i} decompressed: MSE = {mse:.3e}')

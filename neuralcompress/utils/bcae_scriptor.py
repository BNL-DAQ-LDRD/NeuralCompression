"""
Script the BCAE encoder and decoder
"""

import argparse
from pathlib import Path
import torch

from neuralcompress.models.bcae_encoder import BCAEEncoder
from neuralcompress.models.bcae_decoder import BCAEDecoder


def locate(path, epoch):
    """
    Sometimes, we may save model pt files with
    zero-filled epoch number for correct file
    listing in linux file system.
    Hence, we may not be able to locate the
    models simply by epoch number.
    So, we make this function to locate the
    model files name by epoch.
    """
    path = Path(path)
    encoder_fnames = path.glob('encoder*')
    decoder_fnames = path.glob('decoder*')

    encoder_fname = [f for f in encoder_fnames if int(f.stem.split('_')[-1]) == epoch]
    decoder_fname = [f for f in decoder_fnames if int(f.stem.split('_')[-1]) == epoch]

    assert len(encoder_fname) == 1, "non existent or ambiguous encoder filename"
    assert len(decoder_fname) == 1, "non existent or ambiguous decoder filename"

    return encoder_fname[0], decoder_fname[0]


def main():
    """
    main
    """
    parser = argparse.ArgumentParser(
        description="script the BCAE encoder and decoder"
    )

    parser.add_argument(
        '--checkpoint_path',
        '-c',
        required=True,
        type=str,
        help="The path to the checkpoints."
    )

    parser.add_argument(
        '--epoch',
        '-e',
        required=True,
        type=int,
        help="The epoch to load."
    )

    parser.add_argument(
        '--save_path',
        '-s',
        required=True,
        type=str,
        help="The path to save the scripted encoder and decoder."
    )

    parser.add_argument(
        '--prefix',
        '-p',
        default='bcae',
        required=False,
        type=str,
        help="Prefix to the filename of the scripted encoder and decoder."
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path)
    assert checkpoint_path.exists(), f'{checkpoint_path} does not exist!'

    checkpoint_path.glob('encoder_*')

    epoch = args.epoch
    encoder_fname, decoder_fname = locate(checkpoint_path, epoch)

    print(encoder_fname)
    print(decoder_fname)
    assert encoder_fname.exists(), f'{encoder_fname} does not exist!'
    assert decoder_fname.exists(), f'{decoder_fname} does not exist!'

    # load encoder and decoder and load the trained weights
    encoder = BCAEEncoder()
    decoder = BCAEDecoder()
    encoder.load_state_dict(torch.load(encoder_fname))
    decoder.load_state_dict(torch.load(decoder_fname))
    encoder_scripted = torch.jit.script(encoder)
    decoder_scripted = torch.jit.script(decoder)

    # script and save the trained model
    save_path = Path(args.save_path)
    if not save_path.exists():
        choice = input(f'{str(save_path)} does not exists. Create? [Y/n]')
        if choice == 'Y':
            save_path.mkdir(parents=True)

    prefix = args.prefix
    scripted_encoder_fname = f'{save_path}/{prefix}_encoder.pt'
    scripted_decoder_fname = f'{save_path}/{prefix}_decoder.pt'
    encoder_scripted.save(scripted_encoder_fname)
    decoder_scripted.save(scripted_decoder_fname)

    print(f'scripted BCAE encoder saved to: {scripted_encoder_fname}')
    print(f'scripted BCAE decoder saved to: {scripted_decoder_fname}')


if __name__ == '__main__':
    main()

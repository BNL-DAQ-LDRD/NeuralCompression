"""
Script the BCAE encoder and decoder
"""

import argparse
from pathlib import Path
import torch

from load_bcae_models import load_bcae_encoder, load_bcae_decoder


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

    epoch = args.epoch
    encoder = load_bcae_encoder(checkpoint_path, epoch)
    decoder = load_bcae_decoder(checkpoint_path, epoch)
    encoder_scripted = torch.jit.script(encoder)
    decoder_scripted = torch.jit.script(decoder)

    # script and save the trained model
    save_path = Path(args.save_path)
    if not save_path.exists():
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

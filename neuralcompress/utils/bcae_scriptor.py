"""
Script the BCAE encoder and decoder
"""

import argparse
from pathlib import Path
import torch

from neuralcompress.models.bcae_encoder import BCAEEncoder
from neuralcompress.models.bcae_decoder import BCAEDecoder



def main():
    """
    main
    """
    parser = argparse.ArgumentParser(
        description="script the BCAE encoder and decoder"
    )
    parser.add_argument('path', type=str)
    parser.add_argument(
        '--prefix',
        '-p',
        default='bcae',
        required=False,
        type=str,
        help="prefix to the filename of the encoder decoder"
    )
    args = parser.parse_args()

    path = Path(args.path)

    if not path.exists():
        choice = input(f'{str(path)} does not exists. Create? [Y/n]')
        if choice == 'Y':
            path.mkdir(parents=True)


    # load encoder and decoder
    encoder = BCAEEncoder()
    decoder = BCAEDecoder()

    prefix = args.prefix
    encoder_scripted = torch.jit.script(encoder)
    decoder_scripted = torch.jit.script(decoder)
    encoder_fname = f'{path}/{prefix}_encoder.pt'
    decoder_fname = f'{path}/{prefix}_decoder.pt'
    encoder_scripted.save(encoder_fname)
    decoder_scripted.save(decoder_fname)

    print(f'BCAE encoder: {encoder_fname}')
    print(f'BCAE decoder: {decoder_fname}')


if __name__ == '__main__':
    main()

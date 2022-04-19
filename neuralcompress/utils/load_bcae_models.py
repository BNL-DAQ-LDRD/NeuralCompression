"""
Load BCAE model
"""
from pathlib import Path
import torch

from neuralcompress.models.bcae_encoder import BCAEEncoder
from neuralcompress.models.bcae_decoder import BCAEDecoder


def locate(path, epoch, model_type):
    """
    Sometimes, we may save model pt files with
    zero-filled epoch number for correct file
    listing in linux file system.
    Hence, we may not be able to locate the
    models simply by epoch number.
    So, we make this function to locate the
    model files name by epoch.
    """
    path         = Path(path)
    model_fnames = path.glob(f'{model_type}*')
    model_fname  = [f for f in model_fnames if f.stem.split('_')[-1] == str(epoch)]

    assert len(model_fname) == 1, "non existent or ambiguous model filename"

    return model_fname[0]


def load_bcae_model(checkpoint_path, epoch, model_type):
    """
    load BCAE encoder or decoder
    """

    assert model_type in ['encoder', 'decoder'], "model_type can only be either encoder or decoder"

    assert Path(checkpoint_path).exists(), f'{checkpoint_path} does not exist!'
    model_fname = locate(checkpoint_path, epoch, model_type)

    if model_type == 'encoder':
        model = BCAEEncoder()
    else:
        model = BCAEDecoder()
    model.load_state_dict(torch.load(model_fname))

    return model


def load_bcae_encoder(checkpoint_path, epoch):
    """
    Load BCAE encoder
    """
    return load_bcae_model(checkpoint_path, epoch, 'encoder')


def load_bcae_decoder(checkpoint_path, epoch):
    """
    Load BCAE decoder
    """
    return load_bcae_model(checkpoint_path, epoch, 'decoder')

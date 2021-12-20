"""
BCAE model
"""

import torch
from neuralcompress.models.autoencoder_trainer import AutoencoderTrainer
from neuralcompress.models.bcae_encoder import BCAEEncoder
from neuralcompress.models.bcae_decoder import BCAEDecoder
from neuralcompress.models.bcae_loss import BCAELoss


class BCAETrainer(AutoencoderTrainer):
    """
    bcae trainer
    """

    def __init__(self):

        # default settings
        encoder        = BCAEEncoder()
        decoder        = BCAEDecoder()
        loss           = BCAELoss()
        optimizer_info = (torch.optim.AdamW, {'lr' : 0.01})
        scheduler_info = (
            torch.optim.lr_scheduler.StepLR,
            {
                'step_size' : 20,
                'gamma'     : .95,
                'verbose'   : True,
            }
        )
        device = 'cuda'

        super().__init__(
            encoder,
            decoder,
            loss,
            optimizer_info,
            scheduler_info,
            device
        )


if __name__ == '__main__':
    trainer = BCAETrainer()

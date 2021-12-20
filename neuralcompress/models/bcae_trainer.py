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

    # class constant for default setting
    ENCODER        = BCAEEncoder()
    DECODER        = BCAEDecoder()
    LOSS           = BCAELoss()
    OPTIMIZER_INFO = (torch.optim.AdamW, {'lr' : 0.01})
    SCHEDULER_INFO = (
        torch.optim.lr_scheduler.StepLR,
        {
            'step_size' : 20,
            'gamma'     : .95,
            'verbose'   : True,
        }
    )
    DEVICE = 'cuda'

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        encoder        = ENCODER,
        decoder        = DECODER,
        loss           = LOSS,
        optimizer_info = OPTIMIZER_INFO,
        scheduler_info = SCHEDULER_INFO,
        device         = DEVICE
    ):
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

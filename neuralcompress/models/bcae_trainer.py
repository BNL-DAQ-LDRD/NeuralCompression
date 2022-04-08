"""
BCAE model
"""

import torch

from neuralcompress.models.autoencoder_trainer import AutoencoderTrainer
from neuralcompress.models.bcae_encoder import BCAEEncoder
from neuralcompress.models.bcae_decoder import BCAEDecoder
from neuralcompress.models.bcae_loss import BCAELoss

#pylint: disable=too-few-public-methods
class BCAETrainer(AutoencoderTrainer):
    """
    bcae trainer
    """

    def __init__(self):

        # default settings
        encoder        = BCAEEncoder()
        decoder        = BCAEDecoder()
        loss           = BCAELoss()
        optimizer_info = (torch.optim.AdamW, {'lr' : 1e-3})
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

    def pipe(self, input_x, is_train):
        """
        encode -> decode -> get losses
        -> backpropagate error and step optimizer
        Used for training and validation during training.
        """
        self.is_train = is_train
        code          = self.encode(input_x)
        output        = self.decode(code)
        losses        = self.loss(output, input_x)
        loss          = losses['loss']

        # Step optimizer
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {key: val.item() for key, val in losses.items()}


if __name__ == '__main__':
    trainer = BCAETrainer()

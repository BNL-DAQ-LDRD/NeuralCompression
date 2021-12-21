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

        # bcae training uses dynamic coefficient
        # in the linear combination of ckassification
        # and regression loss
        # Specifically, the coefficient of classification loss
        # will be scaled up to match that of regression loss
        # Fidn the formula in the pipe function below.
        self.clf_loss_coef = 0
        self.clf_loss_coef_exp = .5


    def pipe(self, input_x, is_train):
        """
        encode -> decode -> get losses
        -> backpropagate error and step optimizer
        Used for training and validation during training.
        """
        self.is_train  = is_train
        code           = self.encode(input_x)
        output         = self.decode(code)
        losses         = self.loss(output, input_x)

        # update the coefficient for classification loss
        # and get the overall loss
        loss_clf = losses['clf. loss']
        loss_reg = losses['reg. loss']
        exp      = self.clf_loss_coef_exp
        old_coef = self.clf_loss_coef
        new_coef = (loss_reg / loss_clf).item()
        ewm_coef = (exp * old_coef + new_coef) / (exp + 1.)

        self.clf_loss_coef = ewm_coef
        loss               = loss_reg + self.clf_loss_coef * loss_clf
        losses['loss']     = loss

        # Step optimizer
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {key: val.item() for key, val in losses.items()}


if __name__ == '__main__':
    trainer = BCAETrainer()

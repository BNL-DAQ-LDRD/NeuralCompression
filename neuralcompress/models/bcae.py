"""
BCAE model
"""
import torch

from neuralcompression.models.model_base import ModelBase
from neuralcompression.models.bcae_encoder import get_bcae_encoder
from neuralcompression.models.bcae_decoder import get_bcae_decoder
from neuralcompression.models.bcae_losses import get_bcae_loss_metric


class BCAE(ModelBase):
    """
    bcae
    """
    def __init__(
        self,
        optimizer,
        scheduler,
        device
    ):
        encoder     = get_bcae_encoder()
        decoder     = get_bcae_decoder()
        loss_metric = get_bcae_loss_metric()

        super().__init__(
            encoder,
            decoder,
            loss_metric,
            optimizer,
            scheduler,
            device
        )


    def handle_epoch_end(self):
        """
        end of epoch bahavior
        """
        self.metrics = {key: 0 for key in self.metrics_seq}
        self.iters   = 0
        self.scheduler.step()

        self.loss_metric.update(self.metrics)


def get_bcae():
    """
    User's should provide such a function.
    And define parameters here.
    """
    optimizer = {
        'fn'     : torch.optim.AdamW,
        'kwargs' : {'lr' : 0.01}
    }
    scheduler = {
        'fn'     : torch.optim.lr_scheduler.StepLR,
        'kwargs' : {
            'step_size' : 20,
            'gamma'     : .95,
            'verbose'   : True,
        }
    }
    device = 'cuda'

    return BCAE(optimizer, scheduler, device)

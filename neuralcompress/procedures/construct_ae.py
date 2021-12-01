"""
Construct autoencoder model, optimizer, and scheduler
"""
from neuralcompress.models.bcae import BCAE
# from neuralcompress.models.cae import CAE
from neuralcompress.models.losses import BCAELossMetrics

from neuralcompress.procedures.init_network import winit_func
from neuralcompress.torch.select import (
    extract_name_kwargs,
    select_optimizer,
    select_scheduler,
)

def construct_ae(config):
    """
    Construct an auto encoder model
    """

    model_name, model_kwargs = extract_name_kwargs(config['model'])

    assert model_name in ['bcae'], \
        f'{model_name} not implemented'

    if model_name == 'bcae':
        model = BCAE(**model_kwargs)
        loss_metrics = BCAELossMetrics(config['loss'])
    # else:
    #     model = CAE(**model_kwargs

    model.apply(winit_func)
    if config['device'] == 'cuda':
        model = model.cuda()

    optimizer = select_optimizer(model.parameters(), config['optimizer'])
    scheduler = select_scheduler(optimizer, config['scheduler'])

    return model, optimizer, scheduler, loss_metrics

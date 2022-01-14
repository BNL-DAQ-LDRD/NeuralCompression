import copy
import torch
from torch import nn
from torch.optim import lr_scheduler

def extract_name_kwargs(obj):
    if isinstance(obj, dict):
        obj    = copy.copy(obj)
        name   = obj.pop('name')
        kwargs = obj
    else:
        name   = obj
        kwargs = {}

    return (name, kwargs)


def get_norm_layer_fn(norm):
    if norm is None:
        return lambda features : nn.Identity()

    if norm == 'layer':
        return lambda features : nn.LayerNorm((features,))

    if norm == 'batch':
        return nn.BatchNorm3d

    if norm == 'instance':
        return nn.InstanceNorm3d

    raise ValueError("Unknown Layer: '%s'" % norm)


def get_activ_layer(activ):
    name, kwargs = extract_name_kwargs(activ)

    if (name is None) or (name == 'linear'):
        return nn.Identity()

    if name == 'gelu':
        return nn.GELU(**kwargs)

    if name == 'relu':
        return nn.ReLU(**kwargs)

    if name == 'leakyrelu':
        return nn.LeakyReLU(**kwargs)

    if (name == 'sigmoid'):
        return nn.Sigmoid()

    raise ValueError("Unknown activation: '%s'" % name)


def select_optimizer(parameters, optimizer):
    name, kwargs = extract_name_kwargs(optimizer)

    if name == 'AdamW':
        return torch.optim.AdamW(parameters, **kwargs)

    if name == 'Adam':
        return torch.optim.Adam(parameters, **kwargs)

    raise ValueError("Unknown optimizer: '%s'" % name)


def linear_scheduler(
    optimizer,
    epochs_warmup,
    epochs_anneal,
    verbose=True
):

    def lambda_rule(epoch, epochs_warmup, epochs_anneal):
        if epoch < epochs_warmup:
            return 1.0

        return 1.0 - (epoch - epochs_warmup) / (epochs_anneal + 1)

    lr_fn = lambda epoch : lambda_rule(epoch, epochs_warmup, epochs_anneal)

    return lr_scheduler.LambdaLR(optimizer, lr_fn, verbose=verbose)


def select_scheduler(optimizer, scheduler):
    name, kwargs = extract_name_kwargs(scheduler)
    # kwargs['verbose'] = True

    if name == 'linear':
        return linear_scheduler(optimizer, **kwargs)

    if name == 'step':
        return lr_scheduler.StepLR(optimizer, **kwargs)

    if name == 'plateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)

    if name == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)

    if name == 'CosineAnnealingWarmRestarts':
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **kwargs)

    raise ValueError("Unknown scheduler '%s'" % name)

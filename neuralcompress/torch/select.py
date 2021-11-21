import copy
import torch
from torch import nn

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


def select_loss(loss):
    name, kwargs = extract_name_kwargs(loss)

    if name.lower() in [ 'l1', 'mae' ]:
        return nn.L1Loss(**kwargs)

    if name.lower() in [ 'l2', 'mse' ]:
        return nn.MSELoss(**kwargs)

    raise ValueError("Unknown loss: '%s'" % name)

#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler

import numpy as np
import tqdm

from neuralcompress.utils.tpc_dataloader import get_tpc_dataloaders
from neuralcompress.models.bcae import BCAE
from neuralcompress.models.losses import get_tpc_losses


def winit_func(model, init_gain=.2):
    classname = model.__class__.__name__
    if (
        hasattr(model, 'weight') and
        (classname.find('Conv') != -1 or classname.find('Linear') != -1)
    ):
        init.xavier_normal_(model.weight.data, init_gain)


train_loader, valid_loader, test_loader = get_tpc_dataloaders(
    '/data/datasets/sphenix/highest_framedata_3d/outer/',
    batch_size  = 32,
    train_sz    = 960,
    valid_sz    = 320,
    test_sz     = 320,
    is_random   = True
)

# Construct encoder network
conv_layer_1 = {
    'out_channels': 8,
    'kernel_size' : [4, 3, 3],
    'padding'     : [1, 0, 1],
    'stride'      : [2, 2, 1]
}
conv_layer_2 = {
    'out_channels': 16,
    'kernel_size' : [4, 4, 3],
    'padding'     : [1, 1, 1],
    'stride'      : [2, 2, 1]
}
conv_layer_3 = {
    'out_channels': 32,
    'kernel_size' : [4, 4, 3],
    'padding'     : [1, 1, 1],
    'stride'      : [2, 2, 1]
}
conv_layer_4 = {
    'out_channels': 32,
    'kernel_size' : [4, 3, 3],
    'padding'     : [1, 0, 1],
    'stride'      : [2, 2, 1]
}

# Construct decoder network
deconv_layer_1 = {
    'out_channels': 16,
    'kernel_size' : [4, 3, 3],
    'padding'     : [1, 0, 1],
    'stride'      : [2, 2, 1],
    'output_padding': 0
}
deconv_layer_2 = {
    'out_channels': 8,
    'kernel_size' : [4, 4, 3],
    'padding'     : [1, 1, 1],
    'stride'      : [2, 2, 1],
    'output_padding': 0
}
deconv_layer_3 = {
    'out_channels': 4,
    'kernel_size' : [4, 4, 3],
    'padding'     : [1, 1, 1],
    'stride'      : [2, 2, 1],
    'output_padding': 0
}
deconv_layer_4 = {
    'out_channels': 2,
    'kernel_size' : [4, 3, 3],
    'padding'     : [1, 0, 1],
    'stride'      : [2, 2, 1],
    'output_padding': 0
}
conv_args_list = [
    conv_layer_1,
    conv_layer_2,
    conv_layer_3,
    conv_layer_4
]
deconv_args_list = [
    deconv_layer_1,
    deconv_layer_2,
    deconv_layer_3,
    deconv_layer_4
]
bcae = BCAE(
    image_channels   = 1,
    code_channels    = 8,
    conv_args_list   = conv_args_list,
    deconv_args_list = deconv_args_list,
    activ            = {'name': 'leakyrelu', 'negative_slope': .2},
    norm             = 'instance'
).cuda()

bcae.apply(winit_func)


# Generic

def format(num):
    if isinstance(num, int):
        return str(num)
    else:
        if num >= 1000:
            return f'{num:.2f}'
        elif 1 <= num < 1000:
            return f'{num:.4f}'
        else:
            return f'{num:.6f}'


def run_epoch(
    loader,
    model,
    optimizer,
    loss_metrics, # model specific
    progbar_desc=None,
    is_train=True
):
    """
    Run one epoch (generic):
    Input:
        - loader: the input data loader;
        - model: the model;
        - optimizer: the optimizer;
        - loss_and_metric (specific): the function that calculate the loss
            for backpropagation and metrics for exhibition;
        - progbar_desc: progress bar description;
        - is_train: whether it is training;
            If True, backpropogate the loss and step the optimizer.
    Output:
        A dictionary with key from the metrics returned by the
        get_loss_metrics function. The value for each key is an array
        of corresponding values from all batches.
    """
    progbar = tqdm.tqdm(
        desc=progbar_desc,
        total=len(loader),
        dynamic_ncols=True
    )
    device = next(model.parameters()).device

    results = {}
    for batch in loader:
        batch  = batch.to(device)

        # run the model and get loss and metrics
        if is_train:
            output = model(batch)
            loss, metric = loss_metrics(output, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                output = model(batch)
                _, metric = loss_metrics(output, batch)

        # update the results
        for key, val in metric.items():
            if key in results:
                results[key].append(val)
            else:
                results[key] = [val]

        metric_avg = {
            key: format(np.mean(val)) for key, val in results.items()
        }

        progbar.set_postfix(metric_avg, refresh=False)
        progbar.update()

    progbar.close()

    return results


def save(path, model, optimizer, scheduler, epoch=None, zfill_len=0):
    """
    Save model, optimizer, and scheduler
    """
    if epoch:
        epoch_str = str(epoch).zfill(zfill_len)
    else:
        epoch_str = 'final'
    torch.save(model.state_dict(),     f'{path}/mod_{epoch_str}.pt')
    torch.save(optimizer.state_dict(), f'{path}/opt_{epoch_str}.pt')
    torch.save(scheduler.state_dict(), f'{path}/sch_{epoch_str}.pt')


# Model specific arguments
epochs = 40

optimizer = torch.optim.AdamW(bcae.parameters(), lr=0.01)
scheduler = lr_scheduler.StepLR(
    optimizer,
    step_size=20,
    gamma=0.95,
    verbose=True
)

loss_args = {
    'transform'        : lambda x: torch.exp(x) * 6 + 64,
    'weight_pow'       : .1,
    'clf_threshold'    : .5,
    'target_threshold' : 64,
    'gamma'            : 2,
    'eps'              : 1e-8,
    'lambda'           : 20000 # loss = lambda + loss_clf + loss_reg
}

path = ( # must be an absolute path
    '/home/yhuang2/PROJs/NeuralCompression/neuralcompress/'
    'models/train_results/checkpoints'
)
save_freq = 2

def bcae_loss_metrics(output, target, loss_args):
    """
    Get losses and other metrics.
    Input:
    Output:
    """
    output_clf, output_reg = output

    loss_clf, loss_reg = get_tpc_losses(
        output_clf,
        output_reg,
        target,
        loss_args
    )
    loss = loss_reg + loss_args['lambda'] * loss_clf

    metric = nn.MSELoss()
    transform = loss_args['transform']
    threshold = loss_args['clf_threshold']
    output_combined = transform(output_reg) * (output_clf > threshold)
    mse = metric(output_combined, target)

    result = {
        'loss clf': loss_clf.item(),
        'loss reg': loss_reg.item(),
        'loss': loss.item(),
        'mse': mse.item()
    }

    return loss, result


for epoch in range(1, epochs + 1):
    loss_metrics = lambda x, y: bcae_loss_metrics(x, y, loss_args)

    train_results = run_epoch(
        train_loader,
        bcae,
        optimizer,
        loss_metrics = loss_metrics,
        progbar_desc = f'Epoch {epoch}/{epochs} Train',
        is_train     = True
    )

    scheduler.step()

    valid_results = run_epoch(
        valid_loader,
        bcae,
        optimizer,
        loss_metrics = loss_metrics,
        progbar_desc = f'Epoch {epoch}/{epochs} Valid',
        is_train     = False
    )

    loss_avg_reg = np.mean(train_results['loss reg'])
    loss_avg_clf = np.mean(train_results['loss clf'])
    loss_args['lambda'] = loss_avg_reg / loss_avg_clf

    if epoch % save_freq == 0:
        save(path, bcae, optimizer, scheduler, epoch, len(str(epochs)))

save(path, bcae, optimizer, scheduler)

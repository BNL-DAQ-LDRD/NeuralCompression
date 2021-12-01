"""
Generic training functions
"""
import torch

import numpy as np
import tqdm

from neuralcompress.utils.tpc_dataloader import get_tpc_dataloaders
from neuralcompress.procedures.construct_ae import construct_ae


def format_float(num):
    """
    Format scaler print precision.
    """
    if isinstance(num, int):
        return str(num)

    if num >= 1000:
        return f'{num:.2f}'

    if 1 <= num < 1000:
        return f'{num:.4f}'

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
            loss, metrics = loss_metrics.calculate_loss_metrics(output, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                output = model(batch)
                _, metrics = loss_metrics.calculate_loss_metrics(output, batch)

        # update the results
        for key, val in metrics.items():
            if key in results:
                results[key].append(val)
            else:
                results[key] = [val]

        results_avg = {key: np.mean(val) for key, val in results.items()}

        progbar.set_postfix(
            {key: format_float(val) for key, val in results_avg.items()}, 
            refresh=False
        )
        progbar.update()
    progbar.close()
    
    if is_train:
        loss_metrics.update(results_avg)

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


def train(config):
    """
    Construct model and train
    """
    model, optimizer, scheduler, loss_metrics = construct_ae(config)

    train_loader, valid_loader, test_loader = get_tpc_dataloaders(
        config['data_path'],
        **config['data']
    )

    path   = config['save_path']
    epochs = config['epochs']

    for epoch in range(1, epochs + 1):
        train_results = run_epoch(
            train_loader,
            model,
            optimizer,
            loss_metrics = loss_metrics,
            progbar_desc = f'Epoch {epoch}/{epochs} Train',
            is_train     = True
        )

        scheduler.step()

        valid_results = run_epoch(
            valid_loader,
            model,
            optimizer,
            loss_metrics = loss_metrics,
            progbar_desc = f'Epoch {epoch}/{epochs} Valid',
            is_train     = False
        )

        if epoch % config['save_freq'] == 0:
            print(f'Saving model at epoch {epoch}')
            save(path, model, optimizer, scheduler, epoch, len(str(epochs)))

    save(path, model, optimizer, scheduler)

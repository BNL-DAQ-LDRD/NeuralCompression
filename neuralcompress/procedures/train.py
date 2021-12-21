"""
Generic training functions
"""
from collections import defaultdict
import tqdm

from neuralcompress.utils.tpc_dataloader import get_tpc_dataloaders
from neuralcompress.models.bcae_trainer import BCAETrainer


def format_float(num, threshold=6):
    """
    If num is an integer, print the integer.
    If num is a float point number,
        - If the integer part has length over the threshold,
            just print the integer part.
        - If the integer part has length less than or equal
            to the threshold, adjust the precision so that
            totally threshold many number is printed.
    """
    if isinstance(num, int):
        return str(num)

    int_str = str(int(num))
    if len(int_str) > threshold:
        return int_str

    decimal_places = threshold - len(int_str)
    format_string = f'num:.{decimal_places}f'
    return f'{{{format_string}}}'.format(num=num)


def run_epoch(
    loader,
    trainer,
    progbar_desc=None,
    is_train=True
):
    """
    Run one epoch (generic):
    Input:
        - loader: the input data loader;
        - trainer:
        - progbar_desc: progress bar description;
        - is_train: whether it is training;
            If True, backpropogate the loss and step the optimizer.
    Output:
    """
    progbar = tqdm.tqdm(
        desc=progbar_desc,
        total=len(loader),
        dynamic_ncols=True
    )
    device = trainer.device

    # loss_avg = defaultdict(int)
    for i, batch in enumerate(loader):

        loss = trainer.pipe(batch.to(device), is_train)

        # # update the loss average
        # for key, val in loss_dict.items():
        #     loss_avg[key] = (loss_avg[key] * i + val) / (i + 1)

        # update progress bar
        progbar.set_postfix(
            {'loss': loss},
            # {key: format_float(val) for key, val in loss_avg.items()},
            refresh=False
        )
        progbar.update()
    progbar.close()
    trainer.handle_epoch_end()

    # return loss_avg


#pylint:disable=too-many-arguments
def train(
    data_path,
    data_config,
    trainer,
    epochs,
    valid_freq,
    save_path,
    save_freq,
):
    """
    Construct model and train
    """
    train_ldr, valid_ldr, _ = get_tpc_dataloaders(data_path, **data_config)

    epoch_zlen = len(str(epochs))

    for epoch in range(1, epochs + 1):
        descr = f'Epoch {epoch}/{epochs} '
        run_epoch(train_ldr, trainer, f'{descr} Train',True)

        if epoch % valid_freq == 0:
            run_epoch(valid_ldr, trainer, f'{descr} Valid', False)

        if epoch % save_freq == 0:
            print(f'Saving model at epoch {epoch}')
            trainer.save(save_path, epoch, epoch_zlen)

    trainer.save(save_path)


def main():
    """
    main
    """
    trainer = BCAETrainer()

    data_path   = '/data/datasets/sphenix/highest_framedata_3d/outer'
    data_config = {
        'batch_size' : 32,
        'train_sz'   : 960,
        'valid_sz'   : 320,
        'test_sz'    : 320,
        'is_random'  : True,
    }
    epochs      = 2000
    valid_freq  = 5
    save_path   = '/home/yhuang2/PROJs/NeuralCompression_results/checkpoints/'
    save_freq   = 20

    train(
        data_path   = data_path,
        data_config = data_config,
        trainer     = trainer,
        epochs      = epochs,
        valid_freq  = valid_freq,
        save_path   = save_path,
        save_freq   = save_freq
    )


if __name__ == '__main__':
    main()

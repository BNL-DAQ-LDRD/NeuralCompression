"""
base model
"""
import torch
from torch.nn import init

def winit_func(model, init_gain=.2):
    """
    Initialize the network
    Input:
    """
    classname = model.__class__.__name__
    if (
        hasattr(model, 'weight') and
        (classname.find('Conv') != -1 or classname.find('Linear') != -1)
    ):
        init.xavier_normal_(model.weight.data, init_gain)


class AutoencoderTrainer:
    """
    Base Model
    """
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        encoder,
        decoder,
        loss,
        optimizer_info,
        scheduler_info,
        device
    ):
        """
        Initialization
        """

        self.encoder  = encoder
        self.decoder  = decoder
        self.loss     = loss
        self.is_train = True

        self.device = device
        self.encoder.to(device)
        self.decoder.to(device)
        winit_func(self.encoder)
        winit_func(self.decoder)

        optimizer_fn, optimizer_kwargs = optimizer_info
        scheduler_fn, scheduler_kwargs = scheduler_info

        parameters = list(encoder.parameters()) + list(decoder.parameters())
        self.optimizer = optimizer_fn(parameters, **optimizer_kwargs)
        self.scheduler = scheduler_fn(self.optimizer, **scheduler_kwargs)


    def encode(self, input_x):
        """
        Encode
        """
        if not self.is_train:
            with torch.no_grad():
                return self.encoder(input_x)
        return self.encoder(input_x)


    def decode(self, code):
        """
        Decode
        """
        if not self.is_train:
            with torch.no_grad():
                return self.decoder(code)
        return self.decoder(code)


    def pipe(self, input_x, is_train):
        """
        User should define the pipe function
        specifically for its output and loss
        management.
        """
        raise NotImplementedError

    def handle_epoch_end(self):
        """
        Define behavior at the end of each epoch
        """
        self.scheduler.step()


    def save(self, path, epoch=None, zfill_len=0):
        """
        Save models, optimizer, and scheduler
        """
        if epoch:
            epoch_str = str(epoch).zfill(zfill_len)
        else:
            epoch_str = 'final'

        torch.save(self.encoder.state_dict(), f'{path}/encoder_{epoch_str}.pt')
        torch.save(self.decoder.state_dict(), f'{path}/decoder_{epoch_str}.pt')
        torch.save(self.optimizer.state_dict(), f'{path}/opt_{epoch_str}.pt')
        torch.save(self.scheduler.state_dict(), f'{path}/sch_{epoch_str}.pt')

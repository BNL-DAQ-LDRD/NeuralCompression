"""
base model
"""
from collections import defaultdict
import torch

class ModelBase:
    """
    Base Model
    """
    def __init__(
        self,
        encoder,
        decoder,
        loss_metric,
        optimizer,
        scheduler,
        device
    ):
        """
        Initialization
        """

        self.encoder     = encoder
        self.decoder     = decoder
        self.loss_metric = loss_metric
        self.is_train    = True
        self.metrics     = defaultdict(int)
        self.iters       = 0

        self.device = device
        self.encoder.to(device)
        self.decoder.to(device)

        optimizer_fn, optimizer_kwargs = optimizer['fn'], optimizer['kwargs']
        scheduler_fn, scheduler_kwargs = scheduler['fn'], scheduler['kwargs']

        parameters = list(encoder.parameters()) + list(decoder.parameters())
        self.optimizer = optimizer_fn(parameters, **optimizer_kwargs)
        self.scheduler = scheduler_fn(optimizer, **scheduler_kwargs)


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
        encode -> decode -> get loss and metrics
        -> backpropagate error and step optimizer
        Used for training and validation during training.
        """
        self.is_train = is_train
        code          = self.encode(input_x)
        output        = self.decode(code)
        loss, metric  = self.loss_metric(output, input_x)

        # Update metrics record and average
        for key, val in metric.items():
            avg = (self.metrics[key] * self.iters + val) / (self.iters + 1)
            self.metrics[key] = avg
            self.iters += 1

        # Step optimizer
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def handle_epoch_end(self):
        """
        User has to implement the behavior at each epoch end
        """
        raise NotImplementedError


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


    def script(self, path):
        """
        scripting
        """
        encoder_scripted = torch.jit.script(self.encoder)
        decoder_scripted = torch.jit.script(self.decoder)
        encoder_scripted.save(f'{path}/encoder.pt')
        decoder_scripted.save(f'{path}/decoder.pt')

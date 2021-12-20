"""
base model
"""
import torch

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
        encode -> decode -> get losses
        -> backpropagate error and step optimizer
        Used for training and validation during training.
        """
        self.is_train = is_train
        code          = self.encode(input_x)
        output        = self.decode(code)
        loss          = self.loss(output, input_x)

        # Step optimizer
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss


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


    def script(self, path):
        """
        scripting
        """
        encoder_scripted = torch.jit.script(self.encoder)
        decoder_scripted = torch.jit.script(self.decoder)
        encoder_scripted.save(f'{path}/encoder.pt')
        decoder_scripted.save(f'{path}/decoder.pt')

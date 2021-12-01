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

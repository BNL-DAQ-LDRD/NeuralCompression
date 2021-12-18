"""
Define the loss functions
"""
import torch
import torch.nn as nn

class TargetWeightedMSELoss(nn.Module):
    """
    Target weigted MSE loss.
    """
    def __init__(self, weight_pow):
        """
        Input:
            - weight_pow: a positive number
        """
        super().__init__()
        assert weight_pow > 0, \
            'weight_pow must be a postive number!'
        self.weight_pow = weight_pow

    def forward(self, input_x, target):
        """
        Input:
            - input_x: the approximation to the the target tensor.
            - target: the ground truth tensor.
        """
        weight = torch.pow(torch.abs(target), self.weight_pow)
        diff = (input_x - target) ** 2 * weight
        return torch.sum(diff) / torch.sum(weight)


class FocalLoss(nn.Module):
    """
    Focal loss for imbalanced classification.
    From "Focal loss for dense object detection"
    https://arxiv.org/pdf/1708.02002.pdf
    """
    def __init__(self, gamma, eps=1e-8):
        """
        Input:
            - gamma:
            - eps:
        """
        super().__init__()
        self.gamma, self.eps = gamma, eps

    def forward(self, pred, label):
        """
        Input:
        Output:
        """
        p_0, p_1 = 1 - pred + self.eps, pred + self.eps
        l_0, l_1 = ~label, label
        focal_loss = (
            torch.pow(p_1, self.gamma) * torch.log2(p_0) * l_0 +
            torch.pow(p_0, self.gamma) * torch.log2(p_1) * l_1
        )
        return -torch.mean(focal_loss)


def get_transform(transform):
    """
    Get input transform
    """
    if transform is None:
        return nn.Identity()

    if transform == 'bcae':
        return lambda x: torch.exp(x) * 6 + 64

    raise ValueError(f"Unknown transform {transform}")


class LossMetric(nn.Module):
    """
    BCAE loss and metrics class
    """
    def __init__(self, loss_args, metrics):
        """
        Input:
        Output:
        """
        super().__init__()

        self.clf_lambda    = loss_args['lambda']
        self.clf_threshold = loss_args['clf_threshold']
        self.tgt_threshold = loss_args['target_threshold']

        # Classification loss
        gamma = self.loss_args['gamma']
        eps   = self.loss_args['eps']
        if gamma is None:
            self.clf_loss_fn = nn.BCELoss()
        else:
            self.clf_loss_fn = FocalLoss(gamma, eps)

        # Regression loss
        weight_pow = self.loss_args['weight_pow']
        if weight_pow is None:
            self.reg_loss_fn = nn.MSELoss()
        else:
            self.reg_loss_fn = TargetWeightedMSELoss(weight_pow)

        # Metrics
        self.metrics = metrics


    def forward(self, output, target):
        """
        Input:
        Output:
        """
        clf_input, reg_input = output
        result = {}

        # Classification, regression, and total loss
        label = target > self.tgt_threshold
        result['clf. loss'] = self.clf_loss_fn(clf_input, label)

        combined = self.transform(reg_input) * (clf_input > self.clf_threshold)
        result['reg. loss'] = self.reg_loss_fn(combined, target)

        loss = result['reg. loss'] + self.clf_lambda * result['clf. loss']
        result['loss'] = loss

        # Metrics
        for key, metric in self.metrics.items():
            result[key] = metric(combined, target).item()

        return loss, result


    def update(self, metrics):
        """
        Input:
        Output:
        """
        self.clf_lambda = (metrics['reg. loss'] / metrics['clf. loss']).item()


def get_bcae_loss_metric():
    """
    This is a function that user should provide.
    Define the parameters here.
    """

    loss_args = {
        'transform'        : 'bcae',
        'weight_pow'       : .1,
        'clf_threshold'    : .5,
        'target_threshold' : 64,
        'gamma'            : 2,
        'eps'              : 1e-8,
        'lambda'           : 20000, # initial lambda,
        'verbose'          : True,
    },
    metrics = {'mse': nn.MSELoss()}

    return LossMetric(loss_args, metrics)

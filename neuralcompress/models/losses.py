"""
Define the loss functions
"""
import torch
import torch.nn as nn
import numpy as np

def weighted_mse(approx, target, weight=None):
    """
    The weight has to be nonnegative,
    but doesn't have to normalized.
    """
    if weight is None:
        loss = nn.MSELoss()
        return loss(approx, target)
    diff = (approx - target) ** 2 * weight
    return torch.sum(diff) / torch.sum(weight)


def focal_loss(pred, label, gamma=None, eps=1e-8):
    """
    focal loss
    """
    p_0, p_1 = 1 - pred + eps, pred + eps
    l_0, l_1 = ~label, label
    if gamma is None:
        loss = torch.log2(p_1) * label + torch.log2(p_0) * (1 - label)
    else:
        loss = (
            torch.pow(p_1, gamma) * torch.log2(p_0) * l_0 +
            torch.pow(p_0, gamma) * torch.log2(p_1) * l_1
        )
    return -torch.mean(loss)


# class SigmoidStep(nn.Module):
# 	"""
# 	SigmoidStep can also be used to implement a soft-classfication.
# 	"""
# 	def __init__(self, mu, alpha):
# 		super().__init__()
# 		self.mu, self.alpha = mu, alpha
#
# 	def forward(self, x):
# 		y = self.alpha * (x - self.mu)
# 		return torch.sigmoid(y)

def get_transform(transform):
    """
    Get input transform
    """
    if transform is None:
        return nn.Identity()

    if transform == 'bcae':
        return lambda x: torch.exp(x) * 6 + 64

    raise ValueError(f"Unknown transform {transform}")

# pylint: disable=too-many-arguments
def loss_reg(
    clf_input,
    reg_input,
    target,
    transform      = None,
    weight_pow     = None,
    clf_threshold  = .5
):
    """
    Regression loss on the combined output.
    Input:
    Output:
    """
    transform = get_transform(transform)
    transformed = transform(reg_input)

    combined = transformed * (clf_input > clf_threshold)
    if weight_pow is None:
        weight = None
    else:
        weight = torch.pow(torch.abs(target), weight_pow)
    return weighted_mse(combined, target, weight), combined


def loss_clf(
    clf_input,
    target,
    target_threshold = 64,
    gamma            = None,
    eps              = 1e-8
):
    """
    Focal loss for classification.
    From "Focal loss for dense object detection"
    https://arxiv.org/pdf/1708.02002.pdf
    Input:
    Output:
    """
    label = target > target_threshold
    return focal_loss(clf_input, label, gamma, eps=eps)


class BCAELossMetrics():
    """
    BCAE loss and metrics class
    """
    def __init__(self, loss_args):
        """
        Input:
        Output:
        """
        self.loss_args = loss_args
        self.metrics = {'mse': nn.MSELoss()}

    def calculate_loss_metrics(self, output, target):
        """
        Input:
        Output:
        """
        clf_input, reg_input = output
        clf_loss = loss_clf(
            clf_input,
            target,
            target_threshold = self.loss_args['target_threshold'],
            gamma            = self.loss_args['gamma'],
            eps              = self.loss_args['eps']
        )
        reg_loss, combined = loss_reg(
            clf_input,
            reg_input,
            target,
            transform     = self.loss_args['transform'],
            weight_pow    = self.loss_args['weight_pow'],
            clf_threshold = self.loss_args['clf_threshold']
        )
        loss = reg_loss + self.loss_args['lambda'] * clf_loss

        result = {
            'clf. loss': clf_loss.item(),
            'reg. loss': reg_loss.item(),
            'loss': loss.item(),
        }
        for key, metric in self.metrics.items():
            result[key] = metric(combined, target).item()

        return loss, result

    def update(self, metrics):
        """
        Input:
        Output:
        """
        loss_avg_reg = np.mean(metrics['reg. loss'])
        loss_avg_clf = np.mean(metrics['clf. loss'])
        new_lambda = loss_avg_reg / loss_avg_clf
        if self.loss_args['verbose']:
            old_lambda = self.loss_args['lambda']
            print(f'lambda: {old_lambda} -> {new_lambda}')
        self.loss_args['lambda'] = new_lambda

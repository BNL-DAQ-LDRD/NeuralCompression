"""
Define the loss functions
"""
import torch
import torch.nn as nn

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
    if transform is None:
        transformed = reg_input
    else:
        transformed = transform(reg_input)

    combined = transformed * (clf_input > clf_threshold)
    if weight_pow is None:
        weight = None
    else:
        weight = torch.pow(torch.abs(target), weight_pow)
    return weighted_mse(combined, target, weight)


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


def get_tpc_losses(clf_input, reg_input, target, loss_args):
    """
    Get tpc classification and regression losses.
    Input:
    Output:
    """
    clf_loss = loss_clf(
        clf_input,
        target,
        target_threshold = loss_args['target_threshold'],
        gamma            = loss_args['gamma'],
        eps              = loss_args['eps']
    )
    reg_loss = loss_reg(
        clf_input,
        reg_input,
        target,
        transform     = loss_args['transform'],
        weight_pow    = loss_args['weight_pow'],
        clf_threshold = loss_args['clf_threshold']
    )
    return clf_loss, reg_loss

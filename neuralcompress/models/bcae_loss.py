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


class BCAELoss(nn.Module):
    """
    BCAE loss and metrics class
    """

    # class constants for default settings:

    # pylint: disable=too-many-arguments
    def __init__(self):
        """
        Initialize the parameters
        """
        super().__init__()

        # default settings
        transform         = lambda x: torch.exp(x) * 6 + 64
        weight_pow        = .1
        clf_threshold     = .5
        target_threshold  = 64
        gamma             = 2
        eps               = 1e-8
        clf_loss_coef_exp = .5
        other_losses_dict = None

        # set up the network
        self.transform        = transform
        self.target_threshold = target_threshold
        self.clf_threshold    = clf_threshold


        # Classification loss
        if gamma is None:
            self.clf_loss_fn = nn.BCELoss()
        else:
            self.clf_loss_fn = FocalLoss(gamma, eps)

        # Regression loss
        if weight_pow is None:
            self.reg_loss_fn = nn.MSELoss()
        else:
            self.reg_loss_fn = TargetWeightedMSELoss(weight_pow)

        self.clf_loss_coef = 0
        self.clf_loss_coef_exp = clf_loss_coef_exp

        # Additional Loss(es).
        # Given as a dictionary
        if other_losses_dict:
            self.other_losses_dict = other_losses_dict
        else:
            self.other_losses_dict = {}


    def forward(self, output, target):
        """
        Input:
        Output:
        """
        clf_input, reg_input = output
        losses = {}

        # Classification, regression, and total loss
        label = target > self.target_threshold
        losses['clf. loss'] = self.clf_loss_fn(clf_input, label)

        combined = self.transform(reg_input) * (clf_input > self.clf_threshold)
        losses['reg. loss'] = self.reg_loss_fn(combined, target)

        # update coefficient to the classification loss
        self.clf_loss_coef = (self.clf_loss_coef_exp * self.clf_loss_coef \
            + losses['reg. loss'] / losses['clf. loss']) / \
            (1 + self.clf_loss_coef_exp)

        # Get the overall loss
        overall_loss = losses['reg. loss'] \
            + self.clf_loss_coef * losses['clf. loss']
        losses['loss'] = overall_loss

        # in case there are other losses specified
        for loss_type, loss_fn in self.other_losses_dict.items():
            losses[loss_type] = loss_fn(combined, target)

        return losses


if __name__ == '__main__':
    loss = BCAELoss()
    print(loss)

"""
Define the loss functions
"""
import torch
import torch.nn as nn

# def weighted_mse(approx, target, weight=None):
#     """
#     The weight has to be nonnegative,
#     but doesn't have to normalized.
#     """
#     if weight is None:
#         loss = nn.MSELoss()
#         return loss(approx, target)
#     diff = (approx - target) ** 2 * weight
#     return torch.sum(diff) / torch.sum(weight)
#
#
# def focal_loss(pred, label, gamma=None, eps=1e-8):
#     """
#     focal loss
#     """
#     p_0, p_1 = 1 - pred + eps, pred + eps
#     l_0, l_1 = ~label, label
#     if gamma is None:
#         loss = torch.log2(p_1) * label + torch.log2(p_0) * (1 - label)
#     else:
#         loss = (
#             torch.pow(p_1, gamma) * torch.log2(p_0) * l_0 +
#             torch.pow(p_0, gamma) * torch.log2(p_1) * l_1
#         )
#     return -torch.mean(loss)
#
#
# # pylint: disable=too-many-arguments
# def loss_reg(
#     clf_input,
#     reg_input,
#     target,
#     transform      = None,
#     weight_pow     = None,
#     clf_threshold  = .5
# ):
#     """
#     Regression loss on the combined output.
#     Input:
#     Output:
#     """
#     if transform is None:
#         transformed = reg_input
#     else:
#         transformed = transform(reg_input)
#
#     combined = transformed * (clf_input > clf_threshold)
#     if weight_pow is None:
#         weight = None
#     else:
#         weight = torch.pow(torch.abs(target), weight_pow)
#     return weighted_mse(combined, target, weight)
#
#
# def loss_clf(
#     clf_input,
#     target,
#     target_threshold = 64,
#     gamma            = None,
#     eps              = 1e-8
# ):
#     """
#     Focal loss for classification.
#     From "Focal loss for dense object detection"
#     https://arxiv.org/pdf/1708.02002.pdf
#     Input:
#     Output:
#     """
#     label = target > target_threshold
#     return focal_loss(clf_input, label, gamma, eps=eps)
#
# LOSS_ARGS = {
#     'transform'        : lambda x: torch.exp(x) * 6 + 64,
#     'weight_pow'       : .1,
#     'clf_threshold'    : .5,
#     'target_threshold' : 64,
#     'gamma'            : 2,
#     'eps'              : 1e-8,
#     'lambda'           : 20000, # initial lambda,
# }
#
# def get_tpc_losses(output, target, loss_args=LOSS_ARGS):
#     """
#     Get tpc classification and regression losses.
#     Input:
#     Output:
#     """
#     clf_input, reg_input = output
#     clf_loss = loss_clf(
#         clf_input,
#         target,
#         target_threshold = loss_args['target_threshold'],
#         gamma            = loss_args['gamma'],
#         eps              = loss_args['eps']
#     )
#     reg_loss = loss_reg(
#         clf_input,
#         reg_input,
#         target,
#         transform     = loss_args['transform'],
#         weight_pow    = loss_args['weight_pow'],
#         clf_threshold = loss_args['clf_threshold']
#     )
#     return clf_loss, reg_loss


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

    def __call__(self, input_x, target):
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

    def __call__(self, pred, label):
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


    def __call__(self, output, target):
        """
        Input:
        Output:
        """
        clf_input, reg_input = output

        # Classification, regression, and total loss
        label = target > self.target_threshold
        loss_clf = self.clf_loss_fn(clf_input, label)

        combined = self.transform(reg_input) * (clf_input > self.clf_threshold)
        loss_reg = self.reg_loss_fn(combined, target)

        # update coefficient to the classification loss
        exp = self.clf_loss_coef_exp
        old_coef = self.clf_loss_coef
        new_coef = (exp * old_coef + loss_reg / loss_clf) / (1. + exp)
        self.clf_loss_coef = new_coef

        # Get the overall loss
        overall_loss = loss_clf + self.clf_loss_coef * loss_reg

        # save all type of losses to a dictionary
        losses = {}
        losses['clf. loss'] = loss_clf
        losses['reg. loss'] = loss_reg
        losses['loss'] = overall_loss

        # in case there are other losses specified
        for loss_type, loss_fn in self.other_losses_dict.items():
            losses[loss_type] = loss_fn(combined, target)

        return loss_clf, loss_reg


if __name__ == '__main__':
    loss = BCAELoss()
    print(loss)

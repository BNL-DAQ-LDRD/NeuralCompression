"""
combine the classification and regression result
"""
import torch
import torch.nn as nn

class BCAECombine(nn.Module):
    """
    combine the classification and regression result
    """
    def __init__(self):
        super().__init__()

        # default settings
        self.transform     = lambda x: torch.exp(x) * 6 + 64
        self.clf_threshold = .5

    def forward(self, output):
        """
        Input: output from the network,
            - classification result and
            - regression result
        Output:
            combined the result
        """
        clf_input, reg_input = output
        return self.transform(reg_input) * (clf_input > self.clf_threshold)

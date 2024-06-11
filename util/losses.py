import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class InceptionV3Loss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(InceptionV3Loss, self).__init__()
        if smoothing > 0.:
            self.criterion = LabelSmoothingCrossEntropy(smoothing)
        else:
            self.criterion = CrossEntropyLoss()

    def forward(self, input, target):
        # Assuming input is a tuple with main and auxiliary outputs
        # typical for InceptionV3 model.
        if isinstance(input, tuple):
            main_output, aux_output = input
            main_loss = self.criterion(main_output, target)
            aux_loss = self.criterion(aux_output, target)
            # Weighted sum of main loss and auxiliary loss
            # The weight for auxiliary loss is usually set to 0.4
            # Adjust the weights as per your requirement
            return main_loss + 0.4 * aux_loss
        else:
            # If only one output is provided, compute loss directly
            return self.criterion(input, target)

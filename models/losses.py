import torch
from timm.loss import LabelSmoothingCrossEntropy
import numpy as np
from torch import nn
import torch.nn.functional as F
from collections import Counter

class CustomLabelSmoothing(nn.Module):
    def __init__(self,  smoothing=0.1,aux_r=5):
        """
        Constructor for the LabelSmoothingCrossEntropy module.
        :param dataset: Dataset instance to calculate class weights.
        :param smoothing: label smoothing factor
        """
        super(CustomLabelSmoothing, self).__init__()
        assert smoothing < 1.0
        self.aux_r=aux_r
        self.loss_func=LabelSmoothingCrossEntropy(smoothing)
    
    def forward(self, input, target):
        class_tar,patch_tar=target
        class_pred,patch_pred=input
        bc,word_size,num_classes=patch_pred.shape
        patch_tar=patch_tar.reshape(-1)
        patch_pred=patch_pred.reshape(-1,num_classes)
        return self.loss_func(class_pred,class_tar)+self.loss_func(patch_pred,patch_tar)*self.aux_r
    

class AdaptiveCrossEntropyLoss(nn.Module):
    def __init__(self, dataset, device='cpu',aux_r=5., ignore_index=-100):
        """
        Constructor for AdaptiveCrossEntropyLoss.
        :param dataset: Dataset instance to calculate class weights.
        :param device: The device on which the computations are performed.
        :param ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient.
        """
        super(AdaptiveCrossEntropyLoss, self).__init__()
        self.aux_r=aux_r
        self.device = device
        self.class_weights = self.calculate_class_weights(dataset).to(device)
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=ignore_index)
        print("Adaptive loss weight: ",self.class_weights)
        
    def calculate_class_weights(self, dataset):
        # Count the number of occurrences for each class
        label_counter = Counter()
        for _, label,_ in dataset:
            label_counter[label[0]] += 1
        print(label_counter)
        # Calculate weights for classes 1, 2, and 3
        # Set weights for class 0 and 1 to 1.0
        class_weights = [1.0, 1.0]

        # Avoid division by zero and normalize weights for classes 2 and 3
        total_counts_1_2_3 = sum([label_counter[i] for i in range(1, 4)])
        for i in range(2, 4):
            weight = total_counts_1_2_3 / label_counter[i] if label_counter[i] > 0 else 1.0
            class_weights.append(weight)

        # Normalize weights such that the smallest weight is 1.0
        min_weight = min(class_weights)
        class_weights = [w / min_weight for w in class_weights]

        return torch.tensor(class_weights, dtype=torch.float32)

    def forward(self, input, target):
        class_tar,patch_tar=target
        class_pred,patch_pred=input
        bc,word_size,num_classes=patch_pred.shape
        patch_tar=patch_tar.reshape(-1)
        patch_pred=patch_pred.reshape(-1,num_classes)
        return self.cross_entropy_loss(class_pred,class_tar)+self.cross_entropy_loss(patch_pred,patch_tar)*self.aux_r
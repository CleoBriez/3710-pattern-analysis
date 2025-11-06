# recognition\43711451_HipMRI2D_AttentionUNET\utils.py
"""
Contains utility functions for the HipMRI 2D Slice Dataset project
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

__author__ = "Cleodora Kizmann"
__copyright__ = "Copyright 2025, Cleodora Kizmann"
__credits__ = ["Cleodora Kizmann"]
__license__ = "Apache License 2.0"
__version__ = "0.0.1"
__maintainer__ = "Cleodora Kizmann"
__email__ = "cleodora.kizmann@student.uq.edu.au"
__status__ = "Prototype"

# Visualization functions
def denormalize_image(tensor):
    """
    Denormalize a tensor image with mean and std.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    denorm_tensor = tensor * std + mean
    return torch.clamp(denorm_tensor, 0, 1)

class Dice(nn.Module):
    """Dice Loss for binary segmentation.

    Dice Loss = 1 - Dice Coefficient
    Dice Coefficient = (2 * |X ∩ Y|) / (|X| + |Y|)

    Args:
        smooth (float): Smoothing factor to avoid division by zero (default: 1e-6)
    """
    def __init__(self, num_classes = 6, apply_softmax=True, smooth=1e-6):
        super(Dice, self).__init__()
        self.num_classes = num_classes
        self.apply_softmax = apply_softmax
        self.smooth = smooth

    def loss(self, predictions, targets):
        """
        Args:
            predictions: Sigmoid output from model [B, H, W] (values between 0-1)
            targets: Binary ground truth [B, H, W] (values 0 or 1)
        """
        # Apply softmax if predictions are logits
        if self.apply_softmax:
            # Apply softmax across the channel dimension (dim=1)
            predictions = torch.softmax(predictions, dim=1)

        dice_per_class = 0.0

        for i in range(self.num_classes):
            pred_class = predictions[:, i, :, :]
            target_class = targets[:, i, :, :]

            # Flatten tensors using reshape to handle non-contiguous memory layout
            pred_flat = pred_class.reshape(-1)
            target_flat = target_class.reshape(-1)

            # Calculate intersection and union
            intersection = (pred_flat * target_flat).sum()
            dice_sum = pred_flat.sum() + target_flat.sum()

            dice_coeff = (2. * intersection + self.smooth) / (dice_sum + self.smooth)

            dice_per_class += dice_coeff
        
        avg_dice = dice_per_class / self.num_classes

        # Return Dice Loss (1 - Dice Coefficient)
        return 1 - avg_dice
    
    def coeff(self):
        """
        Returns the Dice Coefficient.
        """
        return 1 - self.loss
# recognition\43711451_HipMRI2D_AttentionUNET\utils.py
"""
Contains utility functions for the HipMRI 2D Slice Dataset project
"""

import torch
import torch.nn as nn

__author__ = "Cleodora Kizmann"
__copyright__ = "Copyright 2025, Cleodora Kizmann"
__credits__ = ["Cleodora Kizmann"]
__license__ = "Apache License 2.0"
__version__ = "0.0.1"
__maintainer__ = "Cleodora Kizmann"
__email__ = "cleodora.kizmann@student.uq.edu.au"
__status__ = "Prototype"

class Dice(nn.Module):
    """
    Dice Coefficient = (2 * |X ∩ Y|) / (|X| + |Y|)
    Dice Loss = 1 - Dice Coefficient

    Args:
        num_classes: int, number of classes for segmentation.
        apply_softmax: bool, whether to apply softmax to predictions.
        smooth (float): Smoothing factor to avoid division by zero (default: 1e-6)
    """
    def __init__(self, num_classes = 6, apply_softmax=True, smooth=1e-6):
        super(Dice, self).__init__()
        self.num_classes = num_classes
        self.apply_softmax = apply_softmax
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        Dice Loss calculation.

        Args:
            predictions: Raw logits from model [B, C, H, W]
            targets: One-hot encoded ground truth [B, C, H, W]

        Returns:
            Dice Loss value.
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
    
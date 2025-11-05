# recognition\43711451-HipMRI3D-ImprovedUNET\train.py
"""
Contains the main training script for the model
"""

import utils as utils
import dataset as data
import modules as module
from modules import UNet as model
import torch
import torch.nn as nn
import torch.optim as optim
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

device = utils.device

class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation.

    Dice Loss = 1 - Dice Coefficient
    Dice Coefficient = (2 * |X ∩ Y|) / (|X| + |Y|)

    Args:
        smooth (float): Smoothing factor to avoid division by zero (default: 1e-6)
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Sigmoid output from model [B, H, W] (values between 0-1)
            targets: Binary ground truth [B, H, W] (values 0 or 1)
        """
        # Flatten tensors using reshape to handle non-contiguous memory layout
        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1).float()

        # Calculate intersection and union
        intersection = (predictions * targets).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)

        # Return Dice Loss (1 - Dice Coefficient)
        return 1 - dice_coeff

def show_epoch_predictions(model, dataset, epoch, n=3):
    """Show model predictions after a specific epoch."""
    model.eval()
    fig, axes = plt.subplots(3, n, figsize=(12, 9))
    fig.suptitle(f'🎯 Predictions After Epoch {epoch}', fontsize=16, fontweight='bold')

    with torch.no_grad():
        for i in range(n):
            image, true_mask = dataset[i]

            # Predict with sigmoid model
            pred = model(image.unsqueeze(0).to(device))
            # Get pet class probability and convert to binary
            pred_pet_prob = pred[0, 0].cpu().numpy()  # Pet class probability
            pred_binary = (pred_pet_prob > 0.5).astype(int)  # Binary prediction

            # Denormalize image for visualization
            img_show = module.denormalize_image(image)

            # Show original color image (transpose from CHW to HWC for matplotlib)
            img_display = img_show.permute(1, 2, 0).numpy()  # CHW -> HWC
            axes[0, i].imshow(img_display)
            axes[0, i].set_title(f'Original {i+1}', fontweight='bold')
            axes[0, i].axis('off')

            # Show ground truth binary mask
            axes[1, i].imshow(true_mask, cmap='RdYlBu_r', vmin=0, vmax=1)
            axes[1, i].set_title(f'Ground Truth {i+1}', fontweight='bold')
            axes[1, i].axis('off')

            # Show prediction with accuracy
            axes[2, i].imshow(pred_binary, cmap='RdYlBu_r', vmin=0, vmax=1)
            accuracy = np.mean(pred_binary == true_mask.numpy())
            axes[2, i].set_title(f'Prediction {i+1} (Acc: {accuracy:.3f})', fontweight='bold')
            axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()
    model.train()  # Switch back to training mode
    
def train(model, train_loader, test_dataset, epochs=3, lr=0.001, visualize_every=1):
    model.to(device)
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []

    print(" Starting training with Batch Norm, LeakyReLU, and Sigmoid activation...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Training loop with progress
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            pred_pet = outputs[:, 0]  # Pet class probability from sigmoid
            #print the shape of pred_pet and masks for debugging
            # print(f"pred_pet shape: {outputs.shape}, masks shape: {masks.shape}")
            loss = criterion(pred_pet, masks)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"📈 Epoch {epoch+1}/{epochs} Complete: Avg Loss = {avg_loss:.4f}")

        # Visualize predictions after each epoch (or every few epochs)
        if (epoch) % visualize_every == 0:
            show_epoch_predictions(model, test_dataset, epoch + 1, n=3)

    print(" Training complete with enhanced U-Net!")
    return losses
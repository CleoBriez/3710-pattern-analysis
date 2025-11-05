# recognition\43711451-HipMRI2D-ImprovedUNET\utils.py
"""
Contains utility functions for the HipMRI 2D Slice Dataset project
"""

import torch
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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print(f"Using device: {device}") # It won't recognise my GPU at home :c

# Visualization functions
def denormalize_image(tensor):
    """
    Denormalize a tensor image with mean and std.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    denorm_tensor = tensor * std + mean
    return torch.clamp(denorm_tensor, 0, 1)

def show_examples(dataset, title = "Dataset Examples", n = 3):
    """
    Quick visualization for color demo with binary masks.
    """
    fig, axes = plt.subplots(2, n, figsize=(12, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    for i in range(n):
        image, mask = dataset[i]

        # Denormalize image for visualization
        img_show = denormalize_image(image)

        # Show color image (transpose from CHW to HWC for matplotlib)
        img_display = img_show.permute(1, 2, 0).numpy() # CHW -> HWC
        axes[0, i].imshow(img_display)
        axes[0, i].set_title(f'HipMRI Image {i+1} (Color RGB)', fontweight='bold')
        axes[0, i].axis('off')

        # Debug mask values for this sample
        mask_np = mask.numpy()
        unique_vals = np.unique(mask_np)
        hip_count = np.sum(mask_np == 1)
        bg_count = np.sum(mask_np == 0)

        # Show binary mask with better colormap
        im = axes[1, i].imshow(mask_np, cmap='RdBu', vmin=0, vmax=1)
        axes[1, i].set_title(f'Mask {i+1} (Hip:{hip_count}, BG:{bg_count})', fontweight='bold')
        axes[1, i].axis('off')

        # Add colorbar for the first image to show the scale
        if i == 0:
            from matplotlib.colors import ListedColormap
            colors = ['blue', 'red']  # blue for background (0), red for hip (1)
            cmap = ListedColormap(colors)
            im = axes[1, i].imshow(mask_np, cmap=cmap, vmin=0, vmax=1)
            plt.colorbar(im, ax=axes[1, i], shrink=0.6, ticks=[0, 1], label='0=BG, 1=hip')

    plt.tight_layout()
    plt.show()

def show_epoch_predictions(model, dataset, epoch, n=3):
    """
    Show model predictions after a specific epoch.
    """
    model.eval()
    fig, axes = plt.subplots(3, n, figsize=(12, 9))
    fig.suptitle(f'🎯 Predictions After Epoch {epoch}', fontsize=16, fontweight='bold')

    with torch.no_grad():
        for i in range(n):
            image, true_mask = dataset[i]

            # Predict with sigmoid model
            pred = model(image.unsqueeze(0).to(device))
            # Get hip class probability and convert to binary
            pred_hip_prob = pred[0, 0].cpu().numpy()  # hip class probability
            pred_binary = (pred_hip_prob > 0.5).astype(int)  # Binary prediction

            # Denormalize image for visualization
            img_show = denormalize_image(image)

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

# Quick visualization of loss
def plot_loss(losses, loss_type='dice'):
    """
    
    """
    plt.figure(figsize=(8, 4))
    plt.plot(losses, 'bo-', linewidth=2, markersize=8)

    title_map = {
        'bce': '🔥 Training Loss (BCE)',
        'dice': '🔥 Training Loss (Dice)',
        'combined': '🔥 Training Loss (Combined BCE + Dice)'
    }
    plt.title(title_map.get(loss_type, '🔥 Training Loss'), fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.show()
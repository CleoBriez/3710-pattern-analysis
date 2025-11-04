# recognition\43711451-HipMRI3D-ImprovedUNET\predict.py
"""
Contains the main prediction script for the model after training
"""

import dataset as data
import modules as module

__author__ = "Cleodora Kizmann"
__copyright__ = "Copyright 2025, Cleodora Kizmann"
__credits__ = ["Cleodora Kizmann"]
__license__ = "Apache License 2.0"
__version__ = "0.0.1"
__maintainer__ = "Cleodora Kizmann"
__email__ = "cleodora.kizmann@student.uq.edu.au"
__status__ = "Prototype"

def show_predictions(model, dataset, title="🎯 Binary Segmentation Results (Normalized Color)", n=3):
    """Show model predictions vs ground truth for normalized color-based binary segmentation."""
    model.eval()
    fig, axes = plt.subplots(3, n, figsize=(12, 9))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    with torch.no_grad():
        for i in range(n):
            image, true_mask = dataset[i]

            # Predict with sigmoid model
            pred = model(image.unsqueeze(0).to(device))
            # Get pet class probability and convert to binary
            pred_pet_prob = pred[0, 0].cpu().numpy()  # Pet class probability
            pred_binary = (pred_pet_prob > 0.5).astype(int)  # Binary prediction

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

            # Show prediction
            axes[2, i].imshow(pred_binary, cmap='RdYlBu_r', vmin=0, vmax=1)
            accuracy = np.mean(pred_binary == true_mask.numpy())
            axes[2, i].set_title(f'Prediction {i+1} (Acc: {accuracy:.2f})', fontweight='bold')
            axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()

# Show results
show_predictions(model, test_dataset)
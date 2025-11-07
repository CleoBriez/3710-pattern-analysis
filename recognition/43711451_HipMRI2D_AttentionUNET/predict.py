# recognition/43711451_HipMRI2D_AttentionUNET/predict.py
"""
Contains the main prediction script for the model after training
"""

from pathlib import Path
import torch
import numpy as np
from dataset import standardise, resample_to_img
from modules import AttentionUNet as model 
import nibabel as nib
from nibabel import Nifti1Image
from matplotlib import pyplot as plt

__author__ = "Cleodora Kizmann"
__copyright__ = "Copyright 2025, Cleodora Kizmann"
__credits__ = ["Cleodora Kizmann"]
__license__ = "Apache License 2.0"
__version__ = "0.0.1"
__maintainer__ = "Cleodora Kizmann"
__email__ = "cleodora.kizmann@student.uq.edu.au"
__status__ = "Prototype"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 16 # I got 8GB VRAM on my GPU so I might be pushing this a little
SUBSET = 0
NUM_CLASSES = 6
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
SAVED_MODEL_PATH = "C:/Users/cleod/OneDrive/Documents/Work/UQ/COMP3710/Python Workspace/3710-pattern-analysis/recognition/43711451_HipMRI2D_AttentionUNET/saved_models/full_set_5_epochs.pth"
TEMPLATE_IMG_PATH = "D:/keras_slices_data/keras_slices_train/case_004_week_0_slice_0.nii.gz"
SELECTED_SLICE = "case_040_week_2_slice_0.nii.gz"
INPUT_IMG_PATH = f"D:/keras_slices_data/keras_slices_test/{SELECTED_SLICE}"
OUTPUT_MASK_PATH = f"C:/Users/cleod/OneDrive/Documents/Work/UQ/COMP3710/Python Workspace/3710-pattern-analysis/recognition/43711451_HipMRI2D_AttentionUNET/mask_output/pred_mask_{SELECTED_SLICE}"

def predict(model, image_path, template):
    """
    Runs inference on a single NIfTI image, replicating the training pre-processing.
    
    Args:
        model: The trained Attention U-Net model.
        image_path: Path to the input NIfTI image.
        template: Nifti1Image template for resampling.
    Returns:
        pred_mask_np: NumPy array of the predicted mask.
    """
    
    print(f"Processing: {image_path}")
    
    # Load & Standardise
    nifti_image = standardise(image_path)
    
    # Resample
    resampled_nifti = resample_to_img(nifti_image, 
                                        template, 
                                        interpolation="linear",   
                                        # Suppressing annoying warnings
                                        force_resample=True,
                                        copy_header=True)
    
    # Get Data & Normalize
    image_np = resampled_nifti.get_fdata(caching="unchanged")
    
    # Handle extra dims (from load_data_2D)
    if len(image_np.shape) == 3:
        image_np = image_np[:,:,0] # Shape (H, W)
        
    image_np = image_np.astype(np.float32)

    # Normalize *per image*, just like load_data_2D does
    image_np = (image_np - image_np.mean()) / image_np.std()
    
    # Convert to Tensor
    image_tensor = torch.from_numpy(image_np).float()
    
    # Add channel dim: (H, W) -> (1, H, W)
    image_tensor = image_tensor.unsqueeze(0) 
    
    # Add batch dim: (1, H, W) -> (1, 1, H, W)
    image_tensor = image_tensor.unsqueeze(0)
    
    # Move to device
    image_tensor = image_tensor.to(device)

    # Run Model
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        # Get raw logits [1, 6, H, W]
        logits = model(image_tensor)
        
    # Post-process
    # Apply softmax to get probabilities
    probs = torch.softmax(logits, dim = 1) # dim=1 is the Class dimension
    
    # Get the most likely class index for each pixel
    pred_mask = torch.argmax(probs, dim = 1) # Shape [1, H, W]
    
    # Remove batch dim: [1, H, W] -> [H, W]
    pred_mask = pred_mask.squeeze(0)
    
    # Move to CPU as a NumPy array for saving
    pred_mask_np = pred_mask.cpu().numpy()
    
    # Return the mask AND the spatial info for saving
    return pred_mask_np, resampled_nifti.affine, resampled_nifti.header

def visualize_prediction(image_path, template, pred_mask):
    """
    Visualises the original image, the predicted mask, and an overlay.

    Args:
        image_path: Path to the original image.
        template: Nifti1Image template for resampling.
        pred_mask: NumPy array of the predicted mask.
    """

    # Load & Resample original image
    nii = standardise(image_path)
    resampled_nii = resample_to_img(nii, 
                                    template, 
                                    interpolation="linear",
                                    # Suppressing annoying warnings
                                    force_resample=True,
                                    copy_header=True)
    image_np = resampled_nii.get_fdata(caching="unchanged")
    
    # Handle 3D (H, W, 1) -> 2D (H, W)
    if len(image_np.shape) == 3:
        image_np = image_np[:,:,0]

    # Load mask
    mask_nii = nib.load(pred_mask)
    mask_np = mask_nii.get_fdata()

    plt.figure(figsize=(18, 6))
    plt.suptitle(f"{INPUT_IMG_PATH.partition('D:/keras_slices_data/keras_slices_test/')[2]}", fontsize=16)
    # Original Image
    plt.subplot(1, 3, 1)
    plt.title("MRI Slice")
    plt.imshow(image_np, cmap='gray') # Grayscale for the photo
    plt.axis('off')

    # Predicted Mask
    plt.subplot(1, 3, 2)
    plt.title("Predicted Mask")
    plt.imshow(mask_np, cmap='viridis')
    plt.axis('off')

    # Overlay
    plt.subplot(1, 3, 3)
    plt.title("Mask Overlay on MRI")
    plt.imshow(image_np, cmap='gray')
    
    mask_np_masked = np.ma.masked_where(mask_np == 0, mask_np)
    plt.imshow(mask_np_masked, cmap='viridis', alpha=0.6)
    plt.axis('off')

    plt.tight_layout()
    # Save the figure
    output_fig_path = Path(OUTPUT_MASK_PATH).with_suffix('.png')
    plt.savefig(output_fig_path, dpi=300)
    print(f"Visualization saved to {output_fig_path}")
    plt.show()

if __name__ == "__main__":
    print(f"💛 Initialising the Attention U-Net model from path:{SAVED_MODEL_PATH} on {device} 💛")
    model = model(num_channels = 1, num_classes = NUM_CLASSES)
    print(f"Loading saved weights from {SAVED_MODEL_PATH}...")
    # map_location = device just makes sure it works even if trained on a GPU and are now predicting on a CPU. 
    model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location = device))
    model.to(device)
    model.eval()
    print(f"💚 Model loaded on {device} 💚")

    try:
        print(f"Loading template from {TEMPLATE_IMG_PATH}...")
        template = standardise(TEMPLATE_IMG_PATH)
    except FileNotFoundError:
        print(f"Error: Template image not found at {TEMPLATE_IMG_PATH}")
        exit()

    try:
        pred_mask, affine, header = predict(
            model = model,
            image_path = INPUT_IMG_PATH,
            template = template,
        )

        print(f"Saving mask preditction to {OUTPUT_MASK_PATH}...")

        # Create a new NIfTI object for the mask
        # We use the affine and header from the *resampled input*
        # so the mask perfectly overlays it.
        mask_nii = Nifti1Image(pred_mask.astype(np.int16), affine, header)
        
        # Update header to reflect 2D shape (or 3D with 1 slice)
        mask_nii.header.set_data_shape(pred_mask.shape)
        mask_nii.header.set_data_dtype(np.int16) # Save as integer
        
        nib.save(mask_nii, OUTPUT_MASK_PATH)
        
        print(f"Prediction complete. Mask saved to {OUTPUT_MASK_PATH}")

        visualize_prediction(
            image_path = INPUT_IMG_PATH,
            template = template,
            pred_mask = OUTPUT_MASK_PATH,
        )

    except FileNotFoundError:
        print(f"Error: Input image not found at {INPUT_IMG_PATH}")

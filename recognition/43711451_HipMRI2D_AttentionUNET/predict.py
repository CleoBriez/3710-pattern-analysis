# recognition\43711451_HipMRI2D_AttentionUNET\predict.py
"""
Contains the main prediction script for the model after training
"""


import torch
import numpy as np
from dataset import HipMRI2D, DataLoader, standardise, resample_to_img
from modules import AttentionUNet as model 
from train import device, train
import matplotlib.pyplot as plt
from matplotlib import transforms


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
NUM_CLASSES = 6
LEARNING_RATE = 1e-4
NUM_EPOCHS = 25

def predict(model, image_path, template):
    """
    Runs inference on a single NIfTI image, replicating the training pre-processing.
    
    Args:

    Returns:

    """
    
    print(f"Processing: {image_path}")
    
    # Load & Standardise
    nifti_image = standardise(image_path[0])
    
    # Resample
    resampled_nifti = resample_to_img(nifti_image, template, interpolation="linear")
    
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

if __name__ == "__main__":
    print("💛 Loading training data 💛")
    training_dataset = HipMRI2D(dataset = "train", first_n = 250)
    training_loader = DataLoader(training_dataset, batch_size = BATCH_SIZE, shuffle = True)
    print("💚 Training data loading complete 💚")

    print("💛 Loading validation data 💛")
    validation_dataset = HipMRI2D(dataset = "validate", first_n = 250)
    validation_loader = DataLoader(validation_dataset, batch_size = BATCH_SIZE, shuffle = False)
    print("💚 Validation data loading complete 💚")

    print(f"💛 Initialising the Attention U-Net model on {device} 💛")
    model = model(num_channels = 1 , num_classes = NUM_CLASSES)
    model.to(device)
    print(f"💚 Model initialisation complete on {device} 💚")

    train(training_loader = training_loader, validation_loader = validation_loader, model = model, epochs = NUM_EPOCHS)
    
# more
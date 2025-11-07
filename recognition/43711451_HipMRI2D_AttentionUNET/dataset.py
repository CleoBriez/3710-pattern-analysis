# recognition\43711451_HipMRI2D_AttentionUNET\dataset.py
"""
Contains the data loader and preprocessing for the HipMRI 2D Slice Dataset to be used by the model
"""

import numpy as np
import nibabel as nib
from nibabel import Nifti1Image
from nilearn.image import resample_to_img
from tqdm import tqdm
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

__author__ = "Cleodora Kizmann"
__copyright__ = "Copyright 2025, Cleodora Kizmann"
__credits__ = ["Cleodora Kizmann"]
__license__ = "Apache License 2.0"
__version__ = "0.0.1"
__maintainer__ = "Cleodora Kizmann"
__email__ = "cleodora.kizmann@student.uq.edu.au"
__status__ = "Prototype"

# Dataset path
path = "D:/keras_slices_data/keras_slices_"  # Adjust this path as needed

# Hyperparameters
BATCH_SIZE = 16 # I got 8GB VRAM on my GPU so I might be pushing this a little
SUBSET = 25

def to_channels(arr: np.ndarray, num_classes: int, dtype = np.uint8)-> np.ndarray:
    """
    Converts an integer label array into a one-hot encoded array.
    
    Args:
        arr: The input 2D mask array (H, W).
        num_classes: The total number of classes.
        dtype: The data type of the output array.
        
    Returns:
        A one-hot encoded array of shape (H, W, num_classes).
    """
    res = np.zeros(arr.shape + (num_classes,), dtype = dtype)

    for c in range(num_classes):
        # Set the channel 'c' to 1 where the input array has label 'c'
        res[..., c] = (arr == c)
    return res

def standardise(img_path):
    """
        Helper for if the file is (H, W, 1), it rebuilds it as (H, W, 1).

        Args:
            img_path: Path to the NIfTI image file.
        Returns:
            A Nifti1Image object with standardized dimensions.
    """
    nii = nib.load(img_path)
    
    if len(nii.shape) == 2:
        # Data is 2D (H, W). We need to make it 3D (H, W, 1).
        data_2d = nii.get_fdata(caching = "unchanged") # Shape (H, W)
        data_3d = np.expand_dims(data_2d, axis = -1) # Shape (H, W, 1)
        
        # Re-create the NIfTI object with the new 3D data
        new_nii = Nifti1Image(data_3d, nii.affine, nii.header)
        
        # Manually update the header to reflect the 3D shape
        new_nii.header.set_data_shape(data_3d.shape)
        return new_nii
    elif len(nii.shape) == 3:
        # It's already 3D, just return it.
        return nii
    return nii

# load medical image functions
def load_data_2D(imageNames, normalise = False, categorical = False, num_classes = None, dtype = np.float32, getAffines = False, first_n = 0):
    """
    Load medical image data from names, cases list provided into a list for each.
    Altered to account for slices being different sizes, by resampling to a template image.
    
    Args:
        imageNames: list of paths to NIfTI image files
        normalise: bool (normalise the image 0.0-1.0)
        categorical: bool (If True, 'num_classes' must also be provided)
        num_classes: int (The total number of classes for one-hot encoding, e.g., 6)
        getAffines: bool (Return the affine matrices along with the images)
        first_n: int (Stop loading after n images for quick loading and testing scripts)

    Returns:
        images: np.ndarray of shape (N, H, W) or (N, H, W, C) depending on 'categorical'
        affines: list of affine matrices (if getAffines is True)
    """
    # Validate mask and classes inputs
    if categorical and num_classes is None:
        raise ValueError("You should specify the number of classes when loading categorical mask data.")
    
    affines = [] # Spatial coordinates list

    # Load a template image to get dimensions
    try:
        template_nifti = standardise(imageNames[0])
    except Exception as e:
        print(f"Error loading template image: {imageNames[0]}. {e}")
        return
    
    num = len(imageNames) if first_n == 0 else first_n

    first_case = template_nifti.get_fdata(caching="unchanged")

    if len(first_case.shape) == 3:
        first_case = first_case [:,:,0] # sometimes extra dims, remove to keep 2D slice

    if categorical:
        # first_case = to_channels(first_case, dtype = dtype)
        rows, cols = first_case.shape
        channels = num_classes
        images = np.zeros((num, rows, cols, channels), dtype = dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype = dtype)

    if categorical:
        interpolation = "nearest"  # Preserve integer labels
    else:
        interpolation = "linear"   # Average pixels for smooth image

    for i, inName in enumerate(tqdm(imageNames[:num])):
        niftiImage = standardise(inName) # Loads the image
        # resampled nifti to match template
        resampled_nifti = resample_to_img(
            niftiImage, 
            template_nifti, 
            interpolation = interpolation,
            # Suppressing annoying warnings
            force_resample=True,
            copy_header=True
        )
        # Get data from the *resampled* image
        inImage = resampled_nifti.get_fdata(caching = "unchanged") # read disk only
        affine = resampled_nifti.affine
        if len(inImage.shape) == 3:
            inImage = inImage [:,:,0] # sometimes extra dims in HipMRI_study data 
        inImage = inImage.astype(dtype)

        if normalise and not categorical:
            # ~ inImage = inImage / np.linalg.norm(inImage)
            # # ~ inImage = 255. * inImage / inImage.max () 
            inImage = (inImage - inImage.mean()) / inImage.std() 
        elif(normalise and categorical):
            raise ValueError("You probably didn't mean to normalise categorical mask data.")

        if categorical:
            inImage = to_channels(inImage, num_classes = num_classes, dtype = dtype)
            images[i, :, :, :] = inImage
        else:
            images [i,:,:] = inImage 
        affines.append(affine)

        if first_n != 0 and i == first_n:
            break

    if getAffines:
        return images, affines
    else:
        return images

class HipMRI2D(Dataset):
    """
    Dataset class for segmentation for HipMRI 2D dataset. 

    Args:
        dataset: str, one of "train", "validate", "test" to specify which dataset to load.
        first_n: int, number of samples to load for quick testing (default: 0, load all).

    Returns:
        A PyTorch Dataset object that can be used with DataLoader for training/validation/testing.
    """
    def __init__(self, dataset, first_n = 0):
        """
        Initialize the HipMRI2D dataset.

        Args:
            dataset: str, one of "train", "validate", "test" to specify which dataset to load.
            first_n: int, number of samples to load for quick testing (default: 0, load all).
        """
        self.dataset = load_data_2D(sorted(Path(path + dataset).glob("*.gz")), normalise = True, categorical = False, first_n = first_n) # Shape (N, H, W)
        self.mask_data = load_data_2D(sorted(Path(path + "seg_" + dataset).glob("*.gz")), normalise = False, categorical = True, num_classes = 6, first_n = first_n) # Shape (N, H, W, C)
        self.num_classes = self.mask_data.shape[-1]

        print(f"Image array shape: {self.dataset.shape}") # e.g., (100, 256, 128)
        print(f"Mask array shape: {self.mask_data.shape}")   # e.g., (100, 256, 128, 6)
    
    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.dataset)
    
    def __getitem__(self, index):
        """
        Retrieve the image and corresponding mask at the specified index.

        Args:
            index: Index of the sample to retrieve. 

        Returns:
            A tuple (image, mask) where:
            - image is the preprocessed image tensor.
            - mask is the binary mask tensor for the hip region.
        """
        # Get filename
        image_np = self.dataset[index] # Shape (H, W)
        mask_np = self.mask_data[index] # Shape (H, W, C)

        image_tensor = torch.from_numpy(image_np).float()
        mask_tensor = torch.from_numpy(mask_np).float()

        image_tensor = image_tensor.unsqueeze(0) # (H, W) -> (1, H, W)
        
        # Permute the mask from "channels-last" to "channels-first" becuase PyTorch expects (C, H, W)
        mask_tensor = mask_tensor.permute(2, 0, 1) # (H, W, C) -> (C, H, W)

        return image_tensor, mask_tensor
    
    def get_mean(self):
        return np.mean(self.dataset)
    
    def get_std(self):
        return np.std(self.dataset)
    
class LoadData(DataLoader):
    """
    Custom DataLoader for the HipMRI2D dataset.
    Inherits from torch.utils.data.DataLoader.
    """
    def __init__(self, dataset, first_n = SUBSET, batch_size=BATCH_SIZE, shuffle = True):
        """
        Initialize the DataLoader.

        Args:
            dataset: An instance of the HipMRI2D dataset.
            batch_size: Number of samples per batch to load (default: 16).
            shuffle: Whether to shuffle the data at every epoch (default: True).
        """
        super().__init__(dataset=HipMRI2D(dataset, first_n = first_n), batch_size=batch_size, shuffle=shuffle)

    def __getitem__(self, dataset = "train"):
        """
        Load the dataset and return a DataLoader.

        Args:
            dataset: str, one of "train", "validate", "test" to specify which dataset to load.

        Returns:
            A DataLoader object for the specified dataset.
        """
        retrieved_dataset = HipMRI2D(dataset, first_n = SUBSET)
        loaded_data = DataLoader(retrieved_dataset, batch_size = BATCH_SIZE, shuffle = True)

        return loaded_data

if __name__ == "__main__":
    print("💛 Loading training data 💛")
    LoadData(dataset = "train", first_n = SUBSET, batch_size = BATCH_SIZE, shuffle = True)
    print("💚 Training data loading complete 💚")

    print("💛 Loading validation data 💛")
    LoadData(dataset = "validate", first_n = SUBSET, batch_size = BATCH_SIZE, shuffle = False)
    print("💚 Validation data loading complete 💚")

# recognition\43711451-HipMRI3D-ImprovedUNET\dataset.py
"""
Contains the data loader and preprocessing for the HipMRI 2D Slice Dataset to be used by the model
"""

import utils as util
import numpy as np
import nibabel as nib
from tqdm import tqdm
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

__author__ = "Cleodora Kizmann"
__copyright__ = "Copyright 2025, Cleodora Kizmann"
__credits__ = ["Cleodora Kizmann"]
__license__ = "Apache License 2.0"
__version__ = "0.0.1"
__maintainer__ = "Cleodora Kizmann"
__email__ = "cleodora.kizmann@student.uq.edu.au"
__status__ = "Prototype"

path = "D:/keras_slices_data/keras_slices_"  # Adjust this path as needed

def to_channels(arr: np.ndarray, numClasses: int, dtype = np.uint8)-> np.ndarray:
    """
    Converts an integer label array into a one-hot encoded array.
    
    Args:
        arr: The input 2D mask array (H, W).
        num_classes: The total number of classes (e.g., 6).
        dtype: The data type of the output array.
        
    Returns:
        A one-hot encoded array of shape (H, W, num_classes).
    """
    res = np.zeros(arr.shape +(numClasses,), dtype = dtype)

    for c in range(numClasses):
        # Set the channel 'c' to 1 where the input array has label 'c'
        res[..., c] = (arr == c)
    return res

# load medical image functions
def load_data_2D(imageNames, normImage = False, categorical = False, numClasses = None, dtype = np.float32, getAffines = False, early_stop = False):
    """
    Load medical image data from names, cases list provided into a list for each.
    This function pre - allocates 4 D arrays for conv2d to avoid excessive memory usage.
    
    normImage: bool (normalise the image 0.0-1.0)
    categorical: bool (If True, 'num_classes' must also be provided)
    numClasses: int (The total number of classes for one-hot encoding, e.g., 6)
    getAffines: bool (Return the affine matrices along with the images)
    early_stop: bool (Stop loading pre-maturely, for quick loading and testing scripts)
    """

    affines = []

    if categorical and numClasses is None:
        raise ValueError("You must specify the number of classes when 'categorical=True'")

    # get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching = 'unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case [:,:,0] # sometimes extra dims, remove
    if categorical:
        # first_case = to_channels(first_case, dtype = dtype)
        rows, cols = first_case.shape
        channels = numClasses
        images = np.zeros((num, rows, cols, channels), dtype = dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype = dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching = 'unchanged') # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage [:,:,0] # sometimes extra dims in HipMRI_study data 
        inImage = inImage.astype(dtype)
        if normImage:
            # ~ inImage = inImage / np.linalg.norm(inImage )
            # # ~ inImage = 255. * inImage / inImage.max () 
            inImage =(inImage - inImage.mean())/ inImage.std() 
        if categorical:
            inImage = to_channels(inImage, numClasses = numClasses, dtype=dtype)
            if inImage.shape[2] != images.shape[3]:
                 raise ValueError(f"Shape mismatch error on file {inName}. "
                                  f"Got {inImage.shape[2]} channels, "
                                  f"expected {images.shape[3]}.")
            images[i, :, :, :] = inImage
        else:
            images [i,:,:] = inImage 
        
        affines.append(affine)
        if i > 20 and early_stop:
            break
    if getAffines:
        return images, affines
    else:
        return images

def transform(image, size = (256, 256)):
    """
    Resample a 2D image to the target shape using torchvision transforms.
    """
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    image_tensor = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
    resampled_tensor = transform(image_tensor)
    return resampled_tensor.squeeze(0).numpy()  # Remove channel dimension

class HipMRI2D(Dataset):
    """
    Dataset class for segmentation for HipMRI 2D dataset. 

    This dataset assumes: 
        the file structure retrieved from rangpur 
        path adjusted in the global path variable
        image input is "test", "train" or "validate"
    """
    def __init__(self, dataset = "train", transform = None):
        self.dataset = load_data_2D(sorted(Path(path + dataset).glob("*.gz")), normImage = True, categorical = False, early_stop= True)
        self.mask = load_data_2D(sorted(Path(path + "seg_" + dataset).glob("*.gz")), normImage = False, categorical = True, numClasses = 6, early_stop= True)
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        # Get filename
        image = self.dataset
        mask = self.mask
        
        # Transforms
        if self.transform:
            image = transform(image)

        mask = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)(mask)
        mask_np = np.array(mask)  # Convert PIL to numpy array - this preserves [1,2,3]
        binary_mask = np.zeros_like(mask_np, dtype=np.uint8)
        binary_mask[mask_np == 1] = 1  # prostate pixels = 1
        binary_mask[mask_np == 2] = 0  # background pixels = 0
        binary_mask[mask_np == 3] = 0  # border pixels -> background (no ignored pixels)
        binary_mask[mask_np == 4] = 0  # border pixels -> background (no ignored pixels)
        binary_mask[mask_np == 5] = 0  # border pixels -> background (no ignored pixels)
        binary_mask[mask_np == 6] = 0  # border pixels -> background (no ignored pixels)
        
        # Convert to tensor
        binary_mask = torch.from_numpy(binary_mask).long()

        return image, binary_mask
    
Hip = HipMRI2D(dataset = "train", transform = transform)
HipLoader = DataLoader(Hip, batch_size=32, shuffle=True)
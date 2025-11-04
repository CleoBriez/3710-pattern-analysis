# recognition\43711451-HipMRI3D-ImprovedUNET\dataset.py
"""
Contains the data loader and preprocessing for the HipMRI 2D Slice Dataset to be used by the model
"""

import numpy as np
import nibabel as nib
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

__author__ = "Cleodora Kizmann"
__copyright__ = "Copyright 2025, Cleodora Kizmann"
__credits__ = ["Cleodora Kizmann"]
__license__ = "Apache License 2.0"
__version__ = "0.0.1"
__maintainer__ = "Cleodora Kizmann"
__email__ = "cleodora.kizmann@student.uq.edu.au"
__status__ = "Prototype"

path = "D:\keras_slices_data/"

def to_channels(arr: np.ndarray, dtype = np.uint8)-> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape +(len(channels ),), dtype = dtype)
    for c in channels:
        c = int(c)
        res [..., c: c +1][arr == c] = 1
    return res

# load medical image functions
def load_data_2D(imageNames, normImage = False, categorical = False, dtype = np.float32, getAffines = False, early_stop = False):
    """
    Load medical image data from names, cases list provided into a list for each

    This function pre - allocates 4 D arrays for conv2d to avoid excessive memory usage
    
    normImage: bool(normalise the image 0.0 -1.0) 
    early_stop: Stop loading pre - maturely, leaves arrays mostly empty, for quick loading and testing scripts
    """

    affines = []

    # get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching = 'unchanged')
    if len(first_case.shape)== 3:
        first_case = first_case [:,:,0] # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype = dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros(( num, rows, cols, channels ), dtype = dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols),dtype = dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching = 'unchanged') # read disk only
        affine = niftiImage.affine
        if len(inImage.shape)== 3:
            inImage = inImage [:,:,0] # sometimes extra dims in HipMRI_study data 
        inImage = inImage.astype(dtype)
        if normImage:
            # ~ inImage = inImage / np.linalg.norm(inImage )
            # # ~ inImage = 255. * inImage / inImage.max () 
            inImage =(inImage - inImage.mean())/ inImage.std() 
        if categorical:
            inImage = to_channels(inImage, dtype = dtype)
            images [i,:,:,:] = inImage 
        else:
            images [i,:,:] = inImage 
        
        affines.append(affine)
        if i > 20 and early_stop:
            break
        if getAffines:
            return images, affines
        else:
            return images

class HipMRI2D(Dataset):
    """
    Dataset class for segmentation for HipMRI 2D dataset. 

    This dataset assumes: 
        the file structure retrieved from rangpur 
        path adjusted in the global path variable
        image input is "test", "train" or "validate"
    """
    def __init__(self, image, transform = None):
        self.image = image
        self.mask = "seg_" + image
        self.transform = transform

        self.image_files = load_data_2D(sorted(Path(path + "keras_slices_" + image).glob("*.gz")), normImage = True, categorical = False)
        self.mask_files = load_data_2D(sorted(Path(path + "keras_slices_seg_" + image).glob("*.gz")), normImage = True, categorical = False)

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        # Get filename
        image= self.image_files[index]
        mask = self.mask_files[index]

        # Augmentations 
        if self.transform  :
            augmented = self.transform(image = image, mask = mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask
    
# DO THE DATALOADER
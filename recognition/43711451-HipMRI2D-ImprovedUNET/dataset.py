# recognition\43711451-HipMRI3D-ImprovedUNET\dataset.py
"""
Contains the data loader and preprocessing for the HipMRI 2D Slice Dataset to be used by the model
"""
import numpy as np 
import nibabel as nib 
from tqdm import tqdm 

__author__ = "Cleodora Kizmann"
__copyright__ = "Copyright 2025, Cleodora Kizmann"
__credits__ = ["Cleodora Kizmann, "]
__license__ = "Apache License 2.0"
__version__ = "0.0.1"
__maintainer__ = "Cleodora Kizmann"
__email__ = "cleodora.kizmann@student.uq.edu.au"
__status__ = "Prototype"

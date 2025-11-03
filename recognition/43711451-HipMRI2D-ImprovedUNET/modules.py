# recognition\43711451-HipMRI3D-ImprovedUNET\modules.py
"""
Contains the implementation of the Improved UNet segmentation model to be used for training and prediction
"""

import os

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
    print("Warning CUDA not Found. Using CPU")

def encoder(input):
    return  

def decorder(skipList, input):
    return

def bottleneck(input):
    return
    
def UNet():
    return
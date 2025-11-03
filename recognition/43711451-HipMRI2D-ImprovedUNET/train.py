# recognition\43711451-HipMRI3D-ImprovedUNET\train.py
"""
Contains the main training script for the model
"""

import dataset as data
import modules as module
import torch
import numpy as np
from matplotlib import pyplot

__author__ = "Cleodora Kizmann"
__copyright__ = "Copyright 2025, Cleodora Kizmann"
__credits__ = ["Cleodora Kizmann"]
__license__ = "Apache License 2.0"
__version__ = "0.0.1"
__maintainer__ = "Cleodora Kizmann"
__email__ = "cleodora.kizmann@student.uq.edu.au"
__status__ = "Prototype"

l_rate = 0.0001
epochs = 50

path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/"
train_X, validate_X, test_X = data.get_X_data(path)
train_Y, validate_Y, test_Y = data.get_Y_data(path)
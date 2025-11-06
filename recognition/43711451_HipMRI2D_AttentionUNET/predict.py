# recognition\43711451_HipMRI2D_AttentionUNET\predict.py
"""
Contains the main prediction script for the model after training
"""

import utils as util
import dataset as data
import modules as module
import train as trained
import torch
import torch.nn as nn
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

module.model.train()  # Call the train function from the train module
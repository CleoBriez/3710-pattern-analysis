# recognition\43711451-HipMRI3D-ImprovedUNET\train.py
"""
Contains the main training script for the model
"""

import utils as util
import dataset as data
import modules as module
from modules import AttentionUNet as model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

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

# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 8 # I got 8GB VRAM on my GPU
NUM_EPOCHS = 25
NUM_CLASSES = 6

print("💛 Loading training data 💛")
training_dataset = data.HipMRI2D(dataset = "train", first_n= 20)
training_loader = DataLoader(training_dataset, batch_size=8, shuffle=True) 
print("💚 Training data loading complete 💚")

print("💛 Loading validation data 💛")
training_dataset = data.HipMRI2D(dataset = "validate", first_n= 20)
training_loader = DataLoader(training_dataset, batch_size=8, shuffle=True) 
print("💚 Validation data loading complete 💚")

print(f"💛 Initialising the Attention U-Net model on {device} 💛")
model = model(num_channels= 1 , num_classes = 6)
model.to(device)
print(f"💚 Model initialisation complete on {device} 💚")

def train(training_loader = training_loader, 
            epochs = 1,
            model = model, 
            criterion = util.Dice(),
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)):
    """
    Train the Attention U-Net model with Batch Norm, LeakyReLU, and Sigmoid activation.
    """

    losses = []

    print("🤜 Starting training 🤛")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Training loop with progress
        for (images, masks) in enumerate(training_loader):
            images = images.to(device)
            masks = masks.to(device)

            optim.optimizer.zero_grad()
            outputs = model(images)

            outputs = outputs[:, 0]  # "Pet" class probability from sigmoid
            print(f"image shape: {outputs.shape}, mask shape: {masks.shape}")
            loss = criterion.loss(outputs, masks)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(training_loader)
        losses.append(avg_loss)
        print(f"📈 Epoch {epoch+1}/{epochs} Complete: Avg Loss = {avg_loss:.4f}")

        # Validiation here

    print("✅ Training complete with Attetnion U-Net! ✅")
    return losses




    
    
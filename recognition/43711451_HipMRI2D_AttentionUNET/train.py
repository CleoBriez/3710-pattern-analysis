# recognition\43711451_HipMRI2D_AttentionUNET\train.py
"""
Contains the main training script for the model
"""

from utils import Dice
from dataset import HipMRI2D, DataLoader
from modules import AttentionUNet as model
import torch

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


def train(training_loader = None,
            validation_loader = None, 
            epochs = NUM_EPOCHS,
            model = model, 
):
    """
    Train the Attention U-Net model with the training and validation data loaders.

    Args:
        training_loader: DataLoader for training data.
        validation_loader: DataLoader for validation data.
        epochs: Number of training epochs.
        model: The Attention U-Net model to be trained.
    Returns:
        losses: List of average training losses per epoch.
    """

    criterion = Dice()
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

    losses = []

    print("🤜 Starting training 🤛")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        # Training loop with progress
        for images, masks in training_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # print(f" [DEBUG 1] image shape: {outputs.shape}, mask shape: {masks.shape}")
            loss = criterion.loss(outputs, masks)
            # print("[DEBUG 2] Passed loss calculation")

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(training_loader)
        losses.append(avg_loss)

        print(f"▶ Epoch {epoch+1}/{epochs} Complete: Avg Loss = {avg_loss:.4f} ▶️")

        # Start of Validation Loop
        model.eval()   # Set model to evaluation mode
        epoch_loss_eval = 0

        with torch.no_grad(): 
            for images, masks in validation_loader:
                images = images.to(device)
                masks = masks.to(device)

                # Forward pass only
                outputs = model(images)

                loss = criterion.loss(outputs, masks)
                epoch_loss_eval += loss.item()

        avg_loss_eval = epoch_loss_eval / len(validation_loader)

        print(f"📈 Epoch {epoch+1} / {epochs}")
        print(f"📈 Training Loss: {avg_loss:.4f}")
        print(f"📈 Validation Loss: {avg_loss_eval:.4f}")

    print("✅ Training complete with Attention U-Net! ✅")
    print("💖 Thank you for standing at attention 💖")
    return losses

if __name__ == "__main__":
    print("💛 Loading training data 💛")
    training_dataset = HipMRI2D(dataset = "train", first_n = 20)
    training_loader = DataLoader(training_dataset, batch_size = BATCH_SIZE, shuffle = True)
    print("💚 Training data loading complete 💚")

    print("💛 Loading validation data 💛")
    validation_dataset = HipMRI2D(dataset = "validate", first_n= 20)
    validation_loader = DataLoader(validation_dataset, batch_size = BATCH_SIZE, shuffle = False)
    print("💚 Validation data loading complete 💚")

    print(f"💛 Initialising the Attention U-Net model on {device} 💛")
    model = model(num_channels = 1 , num_classes = NUM_CLASSES)
    model.to(device)
    print(f"💚 Model initialisation complete on {device} 💚")

    train(training_loader = training_loader, validation_loader = validation_loader, model=model)
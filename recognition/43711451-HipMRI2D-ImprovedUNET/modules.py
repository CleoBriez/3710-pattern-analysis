# recognition\43711451-HipMRI3D-ImprovedUNET\modules.py
"""
Contains the implementation of the Improved UNet segmentation model to be used for training and prediction
"""

import utils as utils
import dataset as data
import torch
import torch.nn as nn

__author__ = "Cleodora Kizmann"
__copyright__ = "Copyright 2025, Cleodora Kizmann"
__credits__ = ["Cleodora Kizmann"]
__license__ = "Apache License 2.0"
__version__ = "0.0.1"
__maintainer__ = "Cleodora Kizmann"
__email__ = "cleodora.kizmann@student.uq.edu.au"
__status__ = "Prototype"

class DoubleConv(nn.Module):
    """
    (Convolution -> BatchNorm -> ReLU) * 2
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # First convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Second convolution
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class UNet(nn.Module):
    """
    U-Net architecture for image segmentation with Batch Normalization and ReLU activations.
    """

    def __init__(self, n_channels, n_classes):
        """
        
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels  # Input image channels (e.g., 1 for grayscale, 3 for RGB)
        self.n_classes = n_classes    # Output classes (e.g., 1 for binary, 2+ for multiclass)

        # -----------------
        # Encoder (Down Path)
        # -----------------
        # Each 'inc', 'down1', 'down2', etc., is a "step" in the U.
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512)
        self.down4 = DoubleConv(512, 1024) # The bottleneck
        
        # Max pooling for down-sampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # -----------------
        # Decoder (Up Path)
        # -----------------
        # We use ConvTranspose2d for up-sampling.
        # It doubles the H/W and halves the channels.
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512) # 512 (from up) + 512 (from skip) = 1024

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256) # 256 (from up) + 256 (from skip) = 512

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128) # 128 (from up) + 128 (from skip) = 256

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64) # 64 (from up) + 64 (from skip) = 128

        # -----------------
        # Output Layer
        # -----------------
        # Final 1x1 convolution to map to the number of classes
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        """
        
        """
    # x is the input image, e.g., (BatchSize, 3, 256, 256)

    # ----- Encoder -----
    # We save the output of each encoder block to use in the skip connections
        x1 = self.inc(x)     # -> (B, 64, 256, 256)
        x2 = self.pool(x1)   # -> (B, 64, 128, 128)
        x2 = self.down1(x2)  # -> (B, 128, 128, 128)
        
        x3 = self.pool(x2)   # -> (B, 128, 64, 64)
        x3 = self.down2(x3)  # -> (B, 256, 64, 64)
        
        x4 = self.pool(x3)   # -> (B, 256, 32, 32)
        x4 = self.down3(x4)  # -> (B, 512, 32, 32)
        
        x5 = self.pool(x4)   # -> (B, 512, 16, 16)
        x5 = self.down4(x5)  # -> (B, 1024, 16, 16) - This is the bottleneck

        # ----- Decoder -----
        # In each step, we up-sample, concatenate with the skip connection,
        # and then pass through the DoubleConv block.

        # Step 1
        up_x = self.up1(x5)      # Up-sample x5 -> (B, 512, 32, 32)
        # Concatenate along the channel dimension (dim=1)
        # We concatenate up_x with x4 (the skip connection)
        skip_x = torch.cat([up_x, x4], dim=1) # -> (B, 512+512, 32, 32) = (B, 1024, 32, 32)
        x = self.conv1(skip_x)   # -> (B, 512, 32, 32)

        # Step 2
        up_x = self.up2(x)       # -> (B, 256, 64, 64)
        skip_x = torch.cat([up_x, x3], dim=1) # -> (B, 256+256, 64, 64) = (B, 512, 64, 64)
        x = self.conv2(skip_x)   # -> (B, 256, 64, 64)

        # Step 3
        up_x = self.up3(x)       # -> (B, 128, 128, 128)
        skip_x = torch.cat([up_x, x2], dim=1) # -> (B, 128+128, 128, 128) = (B, 256, 128, 128)
        x = self.conv3(skip_x)   # -> (B, 128, 128, 128)
        
        # Step 4
        up_x = self.up4(x)       # -> (B, 64, 256, 256)
        skip_x = torch.cat([up_x, x1], dim=1) # -> (B, 64+64, 256, 256) = (B, 128, 256, 256)
        x = self.conv4(skip_x)   # -> (B, 64, 256, 256)

        # ----- Output -----
        logits = self.outc(x)    # -> (B, n_classes, 256, 256)
        return logits
    
def denormalize_image(tensor):
    """Denormalize a tensor image with ImageNet mean and std."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    denorm_tensor = tensor * std + mean
    return torch.clamp(denorm_tensor, 0, 1)

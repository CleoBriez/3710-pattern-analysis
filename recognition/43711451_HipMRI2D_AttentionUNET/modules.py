# recognition\43711451_HipMRI2D_AttentionUNET\modules.py
"""
Contains the implementation of the Improved UNet segmentation model to be used for training and prediction
"""

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

class AttentionGate(nn.Module):
    """
    Attention Gate for U-Net skip connections.

    Args:
            gating_channels: Number of channels in the gating signal (from decoder).
            skip_channels: Number of channels in the skip connection (from encoder).
            inter_channels: Number of intermediate channels.
    """

    def __init__(self, gating_channels, skip_channels, inter_channels):
        """
        Initialise the Attention Gate.

        Args:
            gating_channels: Number of channels in the gating signal (from decoder).    
            skip_channels: Number of channels in the skip connection (from encoder).
            inter_channels: Number of intermediate channels.
        """
        super(AttentionGate, self).__init__()
        self.gating_signal = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(inter_channels)
        )

        self.skip_connection = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(inter_channels)
        )

        self.attention_map = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size = 1, stride = 1, padding = 0, bias = True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace = True)

    def forward(self, g, x):
        """
        Forward pass of the Attention Gate.

        Args:
            g: Gating signal (from decoder)
            x: Skip connection (from encoder)
        """
        g1 = self.gating_signal(g)
        x1 = self.skip_connection(x)
        attention_map = self.relu(g1 + x1)
        attention_map = self.attention_map(attention_map)
        return x * attention_map

class DoubleConv(nn.Module):
    """
    (Convolution -> BatchNorm -> ReLU) * 2

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        """
        Initialise the Double Convolution block.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
        super().__init__()
        self.double_conv = nn.Sequential(
            # First convolution
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Second convolution
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass of the Double Convolution block.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after two convolutions.
        """
        return self.double_conv(x)
    
class AttentionUNet(nn.Module):
    """
    U-Net architecture for image segmentation with Batch Normalization and ReLU activations.
    Altered to include Attention Gates in skip connections.

    Args:
        num_channels: Number of input image channels (Keeping it at one since the MRIs are going to be greyscale).
        num_classes: Number of output classes (we going 6 for this one).
    """

    def __init__(self, num_channels = 1, num_classes = 6):
        """
        Initialise the Attention U-Net model.

        Args:
            num_channels: Number of input image channels.
            num_classes: Number of output classes.
        """
        super(AttentionUNet, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes 

        # -----------------
        # Encoder (Down Path)
        # -----------------
        # Each 'inc', 'down1', 'down2', etc., is a "step" in the U.
        self.inc = DoubleConv(num_channels, 64)
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512)
        self.down4 = DoubleConv(512, 1024) # The bottleneck
        # Max pooling for down-sampling
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # -----------------
        # Decoder (Up Path)
        # -----------------
        # ConvTranspose2d for up-sampling doubles the H/W and halves the channels.
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size = 2, stride = 2)
        self.conv1 = DoubleConv(1024, 512) # 512 (from up) + 512 (from skip) = 1024

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size = 2, stride = 2)
        self.conv2 = DoubleConv(512, 256) # 256 (from up) + 256 (from skip) = 512

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2)
        self.conv3 = DoubleConv(256, 128) # 128 (from up) + 128 (from skip) = 256

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2)
        self.conv4 = DoubleConv(128, 64) # 64 (from up) + 64 (from skip) = 128

        # -----------------
        # Output Layer
        # -----------------
        # Final 1x1 convolution to map to the number of classes
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

        # --- ADD THE ATTENTION GATES ---
        # gating_channels = channels from decoder (up-sampled)
        # skip_channels = channels from encoder (skip connection)
        # inter_channels = intermediate channels (can be half of skip_channels)
        self.Att1 = AttentionGate(gating_channels = 512, skip_channels = 512, inter_channels = 256)
        self.Att2 = AttentionGate(gating_channels = 256, skip_channels = 256, inter_channels = 128)
        self.Att3 = AttentionGate(gating_channels = 128, skip_channels = 128, inter_channels = 64)
        self.Att4 = AttentionGate(gating_channels = 64, skip_channels = 64, inter_channels = 32)

    def forward(self, x):
        """
        Forward pass of the Attention U-Net model.

        Args:
            x: Input image tensor.

        Returns:
            logits: Output segmentation logits.
        """
    # x is the input image, e.g., (BatchSize, 3, 256, 256)

    # ----- Encoder -----
    # We save the output of each encoder block to use in the skip connections
        x1 = self.inc(x)    # -> (B, 64, 256, 256)
        x2 = self.pool(x1)  # -> (B, 64, 128, 128)
        x2 = self.down1(x2) # -> (B, 128, 128, 128)
        
        x3 = self.pool(x2)  # -> (B, 128, 64, 64)
        x3 = self.down2(x3) # -> (B, 256, 64, 64)
        
        x4 = self.pool(x3)  # -> (B, 256, 32, 32)
        x4 = self.down3(x4) # -> (B, 512, 32, 32)
        
        x5 = self.pool(x4)  # -> (B, 512, 16, 16)
        x5 = self.down4(x5) # -> (B, 1024, 16, 16) - This is the bottleneck

        # ----- Decoder -----

        up_x = self.up1(x5) # (B, 512, H/8, W/8)
        
        # --- ATTENTION GATE ---
        # 'g' is the gating signal from decoder, 'x' is the skip connection
        x4_att = self.Att1(g = up_x, x = x4)
        # --- (End Gate) ---
        
        skip_x = torch.cat([x4_att, up_x], dim = 1) # Concatenate re-weighted skip
        x = self.conv1(skip_x)

        up_x = self.up2(x)  # (B, 256, H/4, W/4)
        
        # --- ATTENTION GATE ---
        x3_att = self.Att2(g = up_x, x = x3)
        # --- (End Gate) ---
        
        skip_x = torch.cat([x3_att, up_x], dim = 1)
        x = self.conv2(skip_x)

        # Step 3
        up_x = self.up3(x)       # (B, 128, H/2, W/2)
        
        # --- ATTENTION GATE ---
        x2_att = self.Att3(g = up_x, x = x2)
        # --- (End Gate) ---
        
        skip_x = torch.cat([x2_att, up_x], dim = 1)
        x = self.conv3(skip_x)
        
        # Step 4
        up_x = self.up4(x)       # (B, 64, H, W)
        
        # --- ATTENTION GATE ---
        x1_att = self.Att4(g = up_x, x = x1)
        # --- (End Gate) ---
        
        skip_x = torch.cat([x1_att, up_x], dim = 1)
        x = self.conv4(skip_x)

        # ----- Output -----
        logits = self.outc(x)
        return logits
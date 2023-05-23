import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import os
import numpy as np


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        # Define the encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Define the decoder layers
        self.decoder = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Forward pass through the encoder
        encoder_outputs = []
        for module in self.encoder:
            x = module(x)
            encoder_outputs.append(x)
        
        # Forward pass through the decoder
        for module in self.decoder:
            x = module(x)
            # Concatenate with the corresponding encoder output
            x = torch.cat([x, encoder_outputs.pop()], dim=1)
        
        return x

# Define the SPE module
class SPEModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPEModule, self).__init__()
        # Define the three-step processing layers
        self.first_order_pooling = nn.MaxPool2d(kernel_size=2)
        self.second_order_pooling = nn.MaxPool2d(kernel_size=2)
        self.statistical_fusion = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pec = nn.Conv2d(out_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        # First-order pooling
        x_first_order = self.first_order_pooling(x)
        
        # Second-order pooling
        x_second_order = self.second_order_pooling(x_first_order)
        
        # Statistical information fusion
        x_fusion = self.statistical_fusion(x_second_order)
        
        # PEC (Pixel-wise Embedded Confidence)
        x_pec = self.pec(x_fusion)
        
        return x_pec

# Define the SegLog model that combines U-Net and SPE module
class SegLog(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegLog, self).__init__()
        self.unet = UNet(in_channels, out_channels)
        self.spe = SPEModule(out_channels, out_channels)
    
    def forward(self, x):
        # Forward pass through U-Net
        x_unet = self.unet(x)
        
        # Forward pass through SPE module
        x_spe = self.spe(x)
        
        # Combine the outputs of U-Net and SPE module
        output = torch.cat((x_unet, x_spe), dim=1)
        
        return output


class SegLogDataset(Dataset):
    def __init__(self, x_dir, y_dir):
        self.x_dir = x_dir
        self.y_dir = y_dir
        self.file_names = sorted(os.listdir(x_dir))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        x_file = os.path.join(self.x_dir, self.file_names[idx])
        y_file = os.path.join(self.y_dir, self.file_names[idx])

        x_data = torch.from_numpy(np.load(x_file).astype(np.float32))
        y_data = np.load(y_file)

        return x_data, y_data

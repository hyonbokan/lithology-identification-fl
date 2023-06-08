import torch
import os
from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

# Define the UNet model
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class SegmentationDataset(Dataset):
    def __init__(self, input_images, label_images, transform=None):
        self.input_images = input_images
        self.label_images = label_images
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, index):
        input_img = self.input_images[index]
        label_img = self.label_images[index]

        if self.transform:
            input_img = self.transform(input_img)
            label_img = self.transform(label_img)

        return input_img, label_img
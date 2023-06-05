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
    def __init__(self, input_dir, label_dir, transform=None):
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.transform = transform
        self.file_list = sorted(os.listdir(input_dir))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        input_path = os.path.join(self.input_dir, self.file_list[index])
        label_path = os.path.join(self.label_dir, self.file_list[index])

        input_image = Image.open(input_path).convert('L')  # Load input image as grayscale
        label_image = Image.open(label_path).convert('L')  # Load label image as grayscale

        if self.transform:
            input_image = self.transform(input_image)
            label_image = self.transform(label_image)

        return input_image, label_image
import torch
import os
from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder
        self.encoder_conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.encoder_relu1 = nn.ReLU()
        self.encoder_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.encoder_relu2 = nn.ReLU()
        self.encoder_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bottleneck_relu = nn.ReLU()

        # Decoder
        self.decoder_upsample1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.decoder_relu1 = nn.ReLU()
        
        self.decoder_upsample2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.decoder_relu2 = nn.ReLU()

        self.decoder_conv3 = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        encoder_out1 = self.encoder_relu1(self.encoder_conv1(x))
        encoder_out2 = self.encoder_relu2(self.encoder_conv2(self.encoder_pool1(encoder_out1)))

        # Bottleneck
        bottleneck_out = self.bottleneck_relu(self.bottleneck_conv(self.encoder_pool2(encoder_out2)))

        # Decoder
        decoder_out1 = self.decoder_relu1(self.decoder_conv1(torch.cat([bottleneck_out, self.decoder_upsample1(bottleneck_out)], dim=1)))
        decoder_out2 = self.decoder_relu2(self.decoder_conv2(torch.cat([encoder_out2, self.decoder_upsample2(decoder_out1)], dim=1)))

        # Output
        output = self.decoder_conv3(decoder_out2)
        return output


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
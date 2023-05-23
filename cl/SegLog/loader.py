from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import os
import numpy as np

class SegLogDataset(Dataset):
    def __init__(self, x_dir, y_dir):
        self.x_dir = x_dir
        self.y_dir = y_dir
        self.transform = ToTensor()
        self.file_names = sorted(os.listdir(x_dir))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        x_file = os.path.join(self.x_dir, self.file_names[idx])
        y_file = os.path.join(self.y_dir, self.file_names[idx])

        x_data = np.load(x_file)
        y_data = np.load(y_file)

        x_data = self.transform(x_data)

        return x_data, y_data

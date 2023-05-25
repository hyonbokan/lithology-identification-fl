import os
import lasio
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

train_dir = '/media/Data-B/my_research/Geoscience_FL/data_well_log/las_files_Lithostrat_data/train'
save_dir = '/media/Data-B/my_research/Geoscience_FL/data_well_log/1D-image-SegLog/train'

log_curves = ['CALI', 'BS', 'DCAL', 'ROP', 'RDEP', 'RSHA', 'RMED', 'SP', 'DTS', 'DTC', 'NPHI', 'GR', 'RHOB', 'DRHO']

lithology_numbers = {30000: 0,
                 65030: 1,
                 65000: 2,
                 80000: 3,
                 74000: 4,
                 70000: 5,
                 70032: 6,
                 88000: 7,
                 86000: 8,
                 99000: 9,
                 90000: 10,
                 93000: 11,
                 12345: 12}


def x_preprocessing(df):
    X = df.fillna(0)
    X = X[log_curves]
    X = X.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    X = X.fillna(0)
    return X

def df_to_img(X):
    # Get the total number of depth points (D) and number of log curves (N)
    D = len(X.index.values)
    N = len(X.columns)
    data = X.to_numpy()
    input_tensor = np.reshape(data, (D, 1, N))
    # Normalize the input tensor
    mean = np.mean(data)
    std = np.std(data)
    input_tensor = (data - mean) / std
    # Reshape the input tensor
    input_tensor = np.reshape(input_tensor, (D, N))
    return input_tensor

def label_to_img(y):
    # Get the total number of depth points (D) and number of lithology classes (C)
    D = len(y)
    C = len(lithology_numbers)
    # Create an empty lithology label image
    label_image = np.zeros((D, C), dtype=np.float32)
    # Set the corresponding class index to 1 for each depth point
    for i in range(D):
        label_image[i, y[i]] = 1
    return label_image

for filename in tqdm(os.listdir(train_dir)):
    if filename.endswith('.las'):
        file_path = os.path.join(train_dir, filename)
        print(f'{file_path} is being processed...')
        las = lasio.read(f'{file_path}')
        df = las.df()
        # print(df.columns)

        # Preprocessing the log curve(input) data
        for curve in log_curves:
            if curve not in df.columns:
                df[curve] = 0
    
        X = x_preprocessing(df)
        input_tensor = df_to_img(X)

        # Convert the input tensor to PIL image
        save_path = os.path.join(save_dir, 'x', f'{os.path.splitext(filename)[0]}.png')
        image = Image.fromarray(input_tensor.astype(np.uint8), mode='L')
        image.save(save_path)

        # Preprocessing labels
        labels = df['FORCE_2020_LITHOFACIES_LITHOLOGY'].fillna(12345).astype(int)

        # Map numeric labels to lithology names
        y = np.array([lithology_numbers[label] for label in labels])

        label_image = label_to_img(y)

        # Convert to PIL image
        save_path_label = os.path.join(save_dir, 'y', f'{os.path.splitext(filename)[0]}.png')
        label_image_pil = Image.fromarray((label_image * 255).astype(np.uint8), mode='L')
        label_image_pil.save(save_path_label)

print('Preprocessing finished')



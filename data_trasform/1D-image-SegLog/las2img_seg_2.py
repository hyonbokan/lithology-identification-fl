import os
import lasio
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

train_dir = '/media/Data-B/my_research/Geoscience_FL/data_well_log/las_files_Lithostrat_data/train'
save_dir = '/media/Data-B/my_research/Geoscience_FL/data_well_log/1D-image-SegLog_DN1/train'

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

def df_to_tensor(X):
    # Get the total number of depth points (D) and number of log curves (N)
    D = len(X.index.values)
    N = len(X.columns)
    data = X.to_numpy()
    input_tensor = np.reshape(data, (D, N, 1))
    return input_tensor

def label_to_tensor(y):
    DL = len(y.index)
    NL = len(y.columns) - 1  # Exclude the label column
    C = 13 # Number of classes
    label_tensor = np.zeros((DL, NL, 13))
    print(label_tensor.shape)
    for i, row in enumerate(label_tensor):
        for j, val in enumerate(row):
            class_label = labels.iloc[i]
            label_tensor[i, j, class_label] = 1
    return label_tensor

for filename in tqdm(os.listdir(train_dir)):
    if filename.endswith('.las'):
        file_path = os.path.join(train_dir, filename)
        print(f'{file_path} is being processed...')
        las = lasio.read(f'{file_path}')
        df = las.df()

        # Preprocessing the log curve(input) data
        for curve in log_curves:
            if curve not in df.columns:
                df[curve] = 0
    
        X = x_preprocessing(df)
        input_tensor = df_to_tensor(X)
        input_tensor = (input_tensor * 255).astype(np.uint8)
        image = Image.fromarray(input_tensor[:, :, 0], mode='L')
        save_path = os.path.join(save_dir, 'x', f'{os.path.splitext(filename)[0]}.png')
        image.save(save_path)

        # Preprocessing labels
        labels = df['FORCE_2020_LITHOFACIES_LITHOLOGY'].fillna(12345).astype(int)
        labels = labels.replace(lithology_numbers)
        y = pd.concat([X, labels], axis=1)

        label_tensor = label_to_tensor(y)
        scaled_label_tensor = (label_tensor[:, :, 0] * 255).astype(np.uint8)

        # Convert to PIL image
        save_path_label = os.path.join(save_dir, 'y', f'{os.path.splitext(filename)[0]}.png')
        label_image = Image.fromarray(scaled_label_tensor, mode='L')
        label_image.save(save_path_label)

print('Preprocessing finished')



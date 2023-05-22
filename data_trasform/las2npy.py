import os
import lasio
import numpy as np
import pandas as pd
from tqdm import tqdm

train_dir = '/media/Data-B/my_research/Geoscience_FL/data_well_log/las_files_Lithostrat_data/test'
save_dir = '/media/Data-B/my_research/Geoscience_FL/data_well_log/1D-image-SegLog/test'

log_curves = ['CALI', 'BS', 'DCAL', 'ROP', 'RDEP', 'RSHA', 'RMED', 'SP', 'DTS', 'DTC', 'NPHI', 'GR', 'RHOB', 'DRHO']

lithology_keys = {30000: 'Sandstone',
                 65030: 'Sandstone/Shale',
                 65000: 'Shale',
                 80000: 'Marl',
                 74000: 'Dolomite',
                 70000: 'Limestone',
                 70032: 'Chalk',
                 88000: 'Halite',
                 86000: 'Anhydrite',
                 99000: 'Tuff',
                 90000: 'Coal',
                 93000: 'Basement',
                 0: 'None'}

# Function to preprocess the LAS data
def preprocess_data(df):
    df_new = df[log_curves]
    X = df_new.fillna(0)
    X.head()
    X = X.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    return X

# Iterate over files in the train directory
for filename in tqdm(os.listdir(train_dir)):
    if filename.endswith('.las'):
        # Read LAS file
        file_path = os.path.join(train_dir, filename)
        print(f'{file_path} is being processed...')
        las = lasio.read(f'{file_path}')
        df = las.df()
        print(df.columns)

        # Add missing log curves and fill with 0
        for curve in log_curves:
            if curve not in df.columns:
                df[curve] = 0

        # Preprocess data
        X = preprocess_data(df)
        D = len(X.index.values)
        N = len(X.columns)
        data = X.to_numpy()
        input_tensor = np.reshape(data, (D, 1, N))
        
        # Save labels
        labels = df['FORCE_2020_LITHOFACIES_LITHOLOGY'].fillna(0)
        lithofaces = [lithology_keys[label] for label in labels]
        y = np.array(lithofaces)
        label_save_path = os.path.join(save_dir, 'y', f'{os.path.splitext(filename)[0]}_labels.npy')
        np.save(label_save_path, y)
        

        # Save input tensor
        save_path = os.path.join(save_dir, 'x', f'{os.path.splitext(filename)[0]}_input.npy')
        np.save(save_path, input_tensor)
        




print("Preprocessing finished!")
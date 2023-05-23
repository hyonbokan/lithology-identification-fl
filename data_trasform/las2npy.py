import os
import lasio
import numpy as np
import pandas as pd
from tqdm import tqdm

train_dir = '/media/Data-B/my_research/Geoscience_FL/data_well_log/las_files_Lithostrat_data/val'
save_dir = '/media/Data-B/my_research/Geoscience_FL/data_well_log/1D-image-SegLog/val'

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


# Iterate over files in the train directory
for filename in tqdm(os.listdir(train_dir)):
    if filename.endswith('.las'):
        # Read LAS file
        file_path = os.path.join(train_dir, filename)
        print(f'{file_path} is being processed...')
        las = lasio.read(f'{file_path}')
        df = las.df()
        # print(df.columns)

        for curve in log_curves:
            if curve not in df.columns:
                df[curve] = 0


        X = df.fillna(0)
        X = X[log_curves]
        X = X.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        X = X.fillna(0)
        D = len(X.index.values)
        N = len(X.columns)
        data = X.to_numpy()
        input_tensor = np.reshape(data, (D, 1, N))
        
        # Save input tensor
        save_path = os.path.join(save_dir, 'x', f'{os.path.splitext(filename)[0]}.npy')
        np.save(save_path, input_tensor)
        
        # Save labels
        labels = df['FORCE_2020_LITHOFACIES_LITHOLOGY'].fillna(12345).astype(int)
        print(labels)
        lithofaces = [lithology_numbers[label] for label in labels]
        y = np.array(lithofaces)
        label_save_path = os.path.join(save_dir, 'y', f'{os.path.splitext(filename)[0]}.npy')
        np.save(label_save_path, y)
        
print("Preprocessing finished!")
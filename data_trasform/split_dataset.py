import shutil
import os
import pandas as pd
from tqdm import tqdm
source_dir = '../../data_well_log/las_files_Lithostrat_data'  # Directory containing the files to be copied
destination_dir = '../../data_well_log/train'  # Directory where the files will be copied
train = pd.read_csv('../../data_well_log/train.csv', sep=';')

# Iterate over each row in the DataFrame
for index, row in tqdm(train.iterrows()):
    filename = row.iloc[0] + '.las'  # Get the value from the first column of the current row
    source_file = os.path.join(source_dir, filename)  # Path of the source file
    destination_file = os.path.join(destination_dir, filename)  # Path of the destination file

    # Copy the file from the source directory to the destination directory
    shutil.copy2(source_file, destination_file)
print("Finished")
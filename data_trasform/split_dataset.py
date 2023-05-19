import os
import random
import shutil


# Set the number of files for each split
train_count = 98
val_count = 10
test_count = 10

# Set the source directory
source_directory = '/media/Data-B/my_research/Geoscience_FL/data_well_log/las_files_Lithostrat_data'

# Set your own directory names for train, validation, and test sets
train_dir = '/media/Data-B/my_research/Geoscience_FL/data_well_log/train/'
val_dir = '/media/Data-B/my_research/Geoscience_FL/data_well_log/val/'
test_dir = '/media/Data-B/my_research/Geoscience_FL/data_well_log/test/'

# Check if the train directory already exists
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
    print("Train directory created.")
else:
    print("Train directory already exists.")

# Check if the val directory already exists
if not os.path.exists(val_dir):
    os.makedirs(val_dir)
    print("Validation directory created.")
else:
    print("Validation directory already exists.")

# Check if the test directory already exists
if not os.path.exists(test_dir):
    os.makedirs(test_dir)
    print("Test directory created.")
else:
    print("Test directory already exists.")

# Get the list of files in the source directory
files = os.listdir(source_directory)

# Shuffle the file list randomly
random.shuffle(files)


# Split the file list into train, val, and test sets
train_files = files[:train_count]
val_files = files[train_count:train_count+val_count]
test_files = files[train_count+val_count:train_count+val_count+test_count]


# Copy train files to train directory
for file_name in train_files:
    src = os.path.join(source_directory, file_name)
    dst = os.path.join(train_dir, file_name)
    shutil.copy(src, dst)

# Copy val files to val directory
for file_name in val_files:
    src = os.path.join(source_directory, file_name)
    dst = os.path.join(val_dir, file_name)
    shutil.copy(src, dst)

# Copy test files to test directory
for file_name in test_files:
    src = os.path.join(source_directory, file_name)
    dst = os.path.join(test_dir, file_name)
    shutil.copy(src, dst)

print("Splitting files into train, val, and test sets completed.")

import os
import random
import shutil
import argparse

parser = argparse.ArgumentParser(description="Split files into train, validation, and test sets.")
parser.add_argument("--source_dir", required=True, help="Path to the source directory containing files.")
parser.add_argument("--train_dir", required=True, help="Path to the train directory.")
parser.add_argument("--val_dir", required=True, help="Path to the validation directory.")
parser.add_argument("--test_dir", required=True, help="Path to the test directory.")
parser.add_argument("--train_count", type=int, default=98, help="Number of files for the train set.")
parser.add_argument("--val_count", type=int, default=10, help="Number of files for the validation set.")
parser.add_argument("--test_count", type=int, default=10, help="Number of files for the test set.")
args = parser.parse_args()

source_directory = args.source_dir
train_dir = args.train_dir
val_dir = args.val_dir
test_dir = args.test_dir
train_count = args.train_count
val_count = args.val_count
test_count = args.test_count

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
    print("Train directory created.")
else:
    print("Train directory already exists.")

if not os.path.exists(val_dir):
    os.makedirs(val_dir)
    print("Validation directory created.")
else:
    print("Validation directory already exists.")

if not os.path.exists(test_dir):
    os.makedirs(test_dir)
    print("Test directory created.")
else:
    print("Test directory already exists.")

files = os.listdir(source_directory)
random.shuffle(files)

train_files = files[:train_count]
val_files = files[train_count:train_count+val_count]
test_files = files[train_count+val_count:train_count+val_count+test_count]

for file_name in train_files:
    src = os.path.join(source_directory, file_name)
    dst = os.path.join(train_dir, file_name)
    shutil.copy(src, dst)

for file_name in val_files:
    src = os.path.join(source_directory, file_name)
    dst = os.path.join(val_dir, file_name)
    shutil.copy(src, dst)

for file_name in test_files:
    src = os.path.join(source_directory, file_name)
    dst = os.path.join(test_dir, file_name)
    shutil.copy(src, dst)

print("Splitting files into train, val, and test sets completed.")

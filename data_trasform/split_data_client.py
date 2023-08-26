import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Split data into client-specific train and test CSV files.")
parser.add_argument("--data_dir", required=True, help="Path to the data directory.")
parser.add_argument("--save_dir", required=True, help="Path to the directory where client-specific files will be saved.")
parser.add_argument("--num_clients", type=int, required=True, help="Number of clients.")
args = parser.parse_args()

data_dir = args.data_dir
save_dir = args.save_dir
num_clients = args.num_clients

train = pd.read_csv(os.path.join(data_dir, 'train.csv'), sep=';')
test = pd.read_csv(os.path.join(data_dir, 'test_with_lables.csv'))

unique_train_wells = train['WELL'].unique()
unique_test_wells = test['WELL'].unique()

train_wells_per_client = len(unique_train_wells) // num_clients
test_wells_per_client = len(unique_test_wells) // num_clients

train_clients = []
start_index = 0
for i in range(num_clients):
    end_index = start_index + train_wells_per_client if i < num_clients - 1 else len(unique_train_wells)
    client_train_wells = unique_train_wells[start_index:end_index]
    client_train_df = train[train['WELL'].isin(client_train_wells)]
    train_clients.append(client_train_df)
    start_index = end_index

test_clients = []
start_index = 0
for i in range(num_clients):
    end_index = start_index + test_wells_per_client if i < num_clients - 1 else len(unique_test_wells)
    client_test_wells = unique_test_wells[start_index:end_index]
    client_test_df = test[test['WELL'].isin(client_test_wells)]
    test_clients.append(client_test_df)
    start_index = end_index

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print('Saving train files...')
for i, client_train_df in enumerate(train_clients):
    file_path = os.path.join(save_dir, f'client_{i+1}_train.csv')
    client_train_df.to_csv(file_path, index=False)
print('Saving test files...')
for i, client_test_df in enumerate(test_clients):
    file_path = os.path.join(save_dir, f'client_{i+1}_test.csv')
    client_test_df.to_csv(file_path, index=False)

print('Files are saved')

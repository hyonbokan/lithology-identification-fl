import os
import pandas as pd


pwd = '/media/Data-B/my_research/Geoscience_FL/data_well_log/'
save_dir = f'{pwd}cl_data/'

num_clients = 2 # hardcode the number of clients


train = pd.read_csv(pwd + 'train.csv', sep=';')
test = pd.read_csv(pwd + 'test_with_lables.csv')

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
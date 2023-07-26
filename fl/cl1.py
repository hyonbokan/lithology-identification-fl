import pandas as pd
import numpy as np
import numpy.random as nr
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from preprocessing import preprocess, show_evaluation
import flwr as fl

train_csv = f'/home/dnlab/Data-B/my_research/Geoscience_FL/data_well_log/cl_data/client_1_train.csv'
test_csv = f'/home/dnlab/Data-B/my_research/Geoscience_FL/data_well_log/cl_data/client_1_test.csv'


train_data = pd.read_csv(train_csv)
test_data = pd.read_csv(test_csv)

lithology_train = train_data['FORCE_2020_LITHOFACIES_LITHOLOGY']
lithology_test = test_data['FORCE_2020_LITHOFACIES_LITHOLOGY']

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
                        93000: 11}

labels_train = lithology_train.map(lithology_numbers)
labels_test = lithology_test.map(lithology_numbers)

# preprocess was changed
train_dataset = preprocess(train_data)
test_data = preprocess(test_data)





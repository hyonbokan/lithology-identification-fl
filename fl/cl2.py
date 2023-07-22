import pandas as pd
import numpy as np
import numpy.random as nr
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from preprocessing import preprocess, show_evaluation
import flwr as fl
import pickle

train_csv = f'/media/Data-B/my_research/Geoscience_FL/data_well_log/cl_data/client_2_train.csv'
test_csv = f'/media/Data-B/my_research/Geoscience_FL/data_well_log/cl_data/client_2_test.csv'

train_data = pd.read_csv(train_csv)
test_data = pd.read_csv(test_csv)

# First preprocessing the data for cl1 - train , test for preprocessing and fit
traindata1, testdata1, lithology1, lithology_numbers = preprocess(train_data, test_data)

#using a 10-fold stratified cross-validation technique and seting the shuffle parameter to true
#as this improved the validation performance bett
split = 10
kf = StratifiedKFold(n_splits=split, shuffle=True)
open_test = np.zeros((len(testdata1), 12))

#100 n-estimators and 10 max-depth
model = XGBClassifier(n_estimators=100, max_depth=10, booster='gbtree',
                      objective='multi:softprob', learning_rate=0.1, random_state=0,
                      subsample=0.9, colsample_bytree=0.9, tree_method='gpu_hist',
                      eval_metric='mlogloss', verbose=2020, reg_lambda=1500)

# Do we need cross validation for fl???
i = 1
for (train_index, test_index) in kf.split(pd.DataFrame(traindata1), pd.DataFrame(lithology1)):
  X_train, X_test = pd.DataFrame(traindata1).iloc[train_index], pd.DataFrame(traindata1).iloc[test_index]
  Y_train, Y_test = pd.DataFrame(lithology1).iloc[train_index], pd.DataFrame(lithology1).iloc[test_index]
#   model.fit(X_train, Y_train, early_stopping_rounds=100, eval_set=[(X_test, Y_test)], verbose=100)
#   prediction = model.predict(X_test)
#   print(show_evaluation(prediction, Y_test))
  i+=1

# Define Flower client; set the variable names of x and y train properly
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        model.fit(traindata1, lithology1)
        return model.get_booster().get_dump()

    def fit(self, parameters, config):
        # Update the model weights
        booster = model.get_booster()
        booster.load_model(parameters)
        
        # Access the train data for the specific client
        client_train_data = traindata1[X_train.index]
        client_lithology = lithology1[X_train.index]
        
        # Fit the model with the client's data
        model.fit(client_train_data, client_lithology, epochs=1)
        
        # Return the updated model weights
        return model.get_booster().get_dump(), len(client_train_data), {}

    def evaluate(self, parameters, config):
        # Update the model weights
        booster = model.get_booster()
        booster.load_model(parameters)
        
        # Access the test data for the specific client
        client_test_data = traindata1[X_test.index]
        client_lithology = lithology1[X_test.index]
        
        # Evaluate the model with the client's data
        loss, accuracy = model.evaluate(client_test_data, client_lithology)
        
        # Return the loss, data count, and accuracy
        return loss, len(client_test_data), {"accuracy": accuracy}
    



# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())




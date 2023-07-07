import pandas as pd
import numpy as np
import numpy.random as nr
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from preprocessing import preprocess, show_evaluation
import pickle

pwd = '/media/Data-B/my_research/Geoscience_FL/data_well_log/cl_data/client_1/'

train_data = f'{pwd}client_1_train.csv'
test_data = f'{pwd}client_1_test.csv'
# First preprocessing the data for cl0 - train , test for preprocessing and fit
traindata1, testdata1, lithology1, lithology_numbers = preprocess(train_data, test_data)

#using a 10-fold stratified cross-validation technique and seting the shuffle parameter to true
#as this improved the validation performance bett
split = 10
kf = StratifiedKFold(n_splits=split, shuffle=True
open_test = np.zeros((len(testdata1), 12))

#100 n-estimators and 10 max-depth
model = XGBClassifier(n_estimators=100, max_depth=10, booster='gbtree',
                      objective='multi:softprob', learning_rate=0.1, random_state=0,
                      subsample=0.9, colsample_bytree=0.9, tree_method='gpu_hist',
                      eval_metric='mlogloss', verbose=2020, reg_lambda=1500)

i = 1
for (train_index, test_index) in kf.split(pd.DataFrame(traindata1), pd.DataFrame(lithology1)):
  X_train, X_test = pd.DataFrame(traindata1).iloc[train_index], pd.DataFrame(traindata1).iloc[test_index]
  Y_train, Y_test = pd.DataFrame(lithology1).iloc[train_index], pd.DataFrame(lithology1).iloc[test_index]
  model.fit(X_train, Y_train, early_stopping_rounds=100, eval_set=[(X_test, Y_test)], verbose=100)
  prediction = model.predict(X_test)
  print(show_evaluation(prediction, Y_test))
  print(f'-----------------------FOLD {i}---------------------')
  i+=
  open_test += model.predict_proba(pd.DataFrame(testdata1))

open_test= pd.DataFrame(open_test/split)
open_test = np.array(pd.DataFrame(open_test).idxmax(axis=1))
print('---------------CROSS VALIDATION COMPLETE')
print('----------------TEST EVALUATION------------------')
return open_test, model, lithology_numbers

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}
    



# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())




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
traindata1, testdata1, lithology1, lithology_numbers = preprocess(train_data, test_data)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        # Initialize the local copy of the model
        self.local_model = XGBClassifier(
            n_estimators=1, max_depth=10, booster='gbtree',
            objective='multi:softprob', learning_rate=0.1, random_state=0,
            subsample=0.9, colsample_bytree=0.9, tree_method='gpu_hist',
            eval_metric='mlogloss', verbose=2020, reg_lambda=1500
        )

    def get_parameters(self, config):
        # Return the model weights
        return self.local_model.get_booster().get_dump()

    def fit(self, parameters, config):
        # Update the local model weights
        booster = self.local_model.get_booster()
        booster.load_model(parameters)
        
        # Access the train data for the specific client
        client_train_data = traindata1
        client_lithology = lithology1
        
        # Fit the model with the client's data
        self.local_model.fit(client_train_data, client_lithology)
        
        # Return the updated model weights
        return self.local_model.get_booster().get_dump(), len(client_train_data), {}

    def evaluate(self, parameters, config):
        # Update the local model weights
        booster = self.local_model.get_booster()
        booster.load_model(parameters)
        
        # Access the test data for the specific client
        client_test_data = testdata1
        client_lithology = lithology_numbers
        
        # Make predictions using the client's data
        predictions = self.local_model.predict(client_test_data)
        
        # Calculate accuracy
        accuracy = accuracy_score(client_lithology, predictions)
        
        # Return the loss, data count, and accuracy
        return None, len(client_test_data), {"accuracy": accuracy}

# Initialize the Flower client
client = FlowerClient()

# Start the Flower client for federated learning
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)




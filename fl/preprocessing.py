import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# loading saved penalty matrix
A = np.load('penalty_matrix.npy')


def drop_columns(data, *args):    
    columns = []
    for _ in args:
        columns.append(_)
        
    data = data.drop(columns, axis=1)
        
    return data


def process(data):
    cols = list(data.columns)
    for _ in cols:  
        data[_] = np.where(data[_] == np.inf, -999, data[_])
        data[_] = np.where(data[_] == np.nan, -999, data[_])
        data[_] = np.where(data[_] == -np.inf, -999, data[_])
        
    return data

def fill_missing_values(df):    
    cols = list(df.columns)
    for _ in cols:

        df[_] = np.where(df[_] == np.inf, -999, df[_])
        df[_] = np.where(df[_] == np.nan, -999, df[_])
        df[_] = np.where(df[_] == -np.inf, -999, df[_])
        
    return df


#Paulo Bestagini's feature augmentation technique from SEG 2016 ML competition
#Link : https://github.com/seg/2016-ml-contest/tree/master/ispl


# Feature windows concatenation function
def augment_features_window(X, N_neig):
    
    # Parameters
    N_row = X.shape[0]
    N_feat = X.shape[1]
 
    # Zero padding
    X = np.vstack((np.zeros((N_neig, N_feat)), X, (np.zeros((N_neig, N_feat)))))
 
    # Loop over windows
    X_aug = np.zeros((N_row, N_feat*(2*N_neig+1)))
    for r in np.arange(N_row)+N_neig:
        this_row = []
        for c in np.arange(-N_neig,N_neig+1):
            this_row = np.hstack((this_row, X[r+c]))
        X_aug[r-N_neig] = this_row
 
    return X_aug
 
# Feature gradient computation function
def augment_features_gradient(X, depth):
    
    # Compute features gradient
    d_diff = np.diff(depth).reshape((-1, 1))
    d_diff[d_diff==0] = 0.001
    X_diff = np.diff(X, axis=0)
    X_grad = X_diff / d_diff
        
    # Compensate for last missing value
    X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    
    return X_grad
 
# Feature augmentation function
def augment_features(X, well, depth, N_neig=1):
    
    # Augment features
    X_aug = np.zeros((X.shape[0], X.shape[1]*(N_neig*2+2)))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]
        X_aug_win = augment_features_window(X[w_idx, :], N_neig)
        X_aug_grad = augment_features_gradient(X[w_idx, :], depth[w_idx])
        X_aug[w_idx, :] = np.concatenate((X_aug_win, X_aug_grad), axis=1)
    
    # Find padded rows
    padded_rows = np.unique(np.where(X_aug[:, 0:7] == np.zeros((1, 7)))[0])
    
    return X_aug, padded_rows

def score(y_true, y_pred):

    '''
    custom metric used for evaluation
    args:
      y_true: actual prediction
      y_pred: predictions made
    '''

    S = 0.0
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    for i in range(0, y_true.shape[0]):
        S -= A[y_true[i], y_pred[i]]
    return S/y_true.shape[0]

def show_evaluation(pred, true):

  '''

  function to show model performance and evaluation
  args:
    pred: predicted value(a list)
    true: actual values (a list)

  prints the custom metric performance, accuracy and F1 score of predictions

  '''

  print(f'Default score: {score(true.values, pred)}')
  print(f'Accuracy is: {accuracy_score(true, pred)}')
  print(f'F1 is: {f1_score(pred, true.values, average="weighted")}')

def preprocess(df):
        
    df_well = df.WELL.values
    df_depth = df.DEPTH_MD.values

    print(f"Shape of concatenated dataframe before dropping columns: {df.shape}")

    cols_to_drop = ['SGR', 'DTS', 'RXO', 'ROPA', 'FORCE_2020_LITHOFACIES_LITHOLOGY', 'FORCE_2020_LITHOFACIES_CONFIDENCE']
    df = df.drop(columns=cols_to_drop, errors='ignore')

    print(f"Shape of dataframe after dropping columns: {df.shape}")
    
    # Label encoding the GROUP, FORMATION, and WELLS features
    df['GROUP_encoded'] = df['GROUP'].astype('category').cat.codes
    df['FORMATION_encoded'] = df['FORMATION'].astype('category').cat.codes
    df['WELL_encoded'] = df['WELL'].astype('category').cat.codes
    print(f"Shape of dataframe after label encoding columns: {df.shape}")

    # Further preparation to split dataframe into train and test datasets after preparation
    df = df.drop(['WELL', 'GROUP', 'FORMATION'], axis=1)

    df = df.fillna(-999)
    df = process(df)

    print(f"Dataframe columns: {df.columns}")
    print(f"Shape of the dataset BEFORE augmentation: {df.shape}")

    augmented_df, _ = augment_features(pd.DataFrame(df).values, df_well, df_depth)

    print(f"Shape of the dataset AFTER augmentation: {augmented_df.shape}")

    return augmented_df


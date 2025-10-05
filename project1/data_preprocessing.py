import numpy as np
from helpers import *
from itertools import combinations_with_replacement

def remove_useless(data: np.ndarray, annotated_data: np.ndarray) -> np.ndarray:
    """Removes useless features from the dataset based on annotated data.

    Args:
        data (np.ndarray): The input data array.
        annotated_data (np.ndarray): The annotated data array indicating useful features.

    Returns:
        np.ndarray: The data array with useless features removed.
    """
    mask = annotated_data[:, 0] != '0'               
    if mask.shape[0] != data.shape[1]:
        raise ValueError(f"Annotation length {mask.shape[0]} != n_features {data.shape[1]}")
    return data[:, mask] 

def read_annotated_csv(path, delimiter=',', skip_header=0 ):
    """Reads a CSV file and returns the data as a NumPy array.

    Args:
        path (str): The path to the CSV file.
        delimiter (str): The delimiter used in the CSV file. Default is ','.
        skip_header (int): The number of header lines to skip. Default is 0.

    Returns:
        np.ndarray: The data from the CSV file as a NumPy array.
    """

    data = np.genfromtxt(
        path,
        delimiter=",",
        skip_header=skip_header,
        dtype=str,            # or None
        autostrip=True,
        comments=None,
        invalid_raise=False,    # don't error on inconsistent rows
        usecols=range(8),       # force 7 columns
        filling_values=np.nan
    )[1:,1:8]

    return data


def separate_cat_and_num(data, data_annoted) :
    """
    Separate categorical and numerical variables from given dataset "data". 
    We have manually created a dataset called data_annoted. This dataset contains all
    the features as rows. It also contains 7 columns (variable (name of the feature), keep (whether we judge the feature is relevant
    or not for predicting heart disease), categorical (1 is categorical 0 is numerical), True/false (whether its a true false categorical variable),
    continuous (1 if continous, 0 if not),"dont know" (specify the different numbers the dontknow or prefer not to say categories that we will replace
    by nan's), special format (whether theres anything special about this feature) and nb_categories which gives the nb of categories in the feature)

    We use those 2 dataset to split our data between numerical and categorical variable. 
    We will use the categorical dataset to one hot encode it. 

    Args : 
        data : the dataset we want to split into numerical and categorical 
        data_annoted : the dataset we created in order to sort the features
    
    Returns : 
        data_categorical : the original dataset that has been amputed from its numerical values -> the indices have changed
        data_numerical : the original dataset that has been amputed from its numerical values -> the indices have changed 
        data_annoted_categorical : the annoted dataset where we removed the numerical values -> need it for no index confusion
    """
    data_annoted = data_annoted[data_annoted[:,0] != '0'] #remove useless
    
    idx_categorical = np.where(data_annoted[:,1] != '0') #gets all categorical indexes
    idx_numerical = np.where(data_annoted[:,1]=='0') #gets all numerical indexes
   
    data_categorical = data[:,idx_categorical[0]] #splits data into only categorical data for later use
    data_numerical = data[:,idx_numerical[0]] #splits data into only numerical data for later use
    
    data_annoted_categorical = data_annoted[idx_categorical,:] #splits the annoted data into only categorical
    data_annoted_categorical = np.squeeze(data_annoted_categorical) #squeeze it bc it created 3d array

    for index in range(len(data_annoted_categorical)): 
        special_vals = data_annoted_categorical[index,4] #get the special values (dont know & prefered not to say and sometimes something else) for each feature : stored in 5th column of our artifically created dataset
        special_val = special_vals.split('&') #split the special values 
        for val in special_val : #iterate through the special values 
            mask = data_categorical[:,index] == val #gets the indices where the feature = special value
            data_categorical[mask,index] = np.nan #replace by NaN
    
    return data_categorical,data_numerical, data_annoted_categorical

def one_hot_encode(data,data_annoted) : 
    """
    Function that one hot encodes (pas le temps de finir d'écrire)
    """
    one_hot_features = []
    one_hot_features = []
    for index in range(len(data_annoted)): 
        n = float(data_annoted[index,6])
        feature = data[:,index]
  
        unique_values = np.arange(n)
        if np.isnan(feature).any():
                categories = np.concatenate([unique_values, [np.nan]])
        else:
                categories = unique_values
    
        one_hot_matrix = np.zeros((data.shape[0], len(categories)))
        
        for i,value in enumerate(feature):
                if np.isnan(value):
                        cat_idx = np.where(np.isnan(categories))[0]
                else:
                        cat_idx = np.where(categories == value)[0]
                one_hot_matrix[i, cat_idx] = 1
        one_hot_features.append(one_hot_matrix)


    onehot = np.hstack(one_hot_features)
    return onehot


def fill_empty_with_nan(data: np.ndarray) -> np.ndarray:
    """Fills empty values in the dataset with NaN.

    Args:
        data (np.ndarray): The input data array.

    Returns:
        np.ndarray: The data array with empty values filled with NaN.
    """
    data_filled = np.where(data == '', np.nan, data)
    return data_filled

def fill_specials(x_train, annotated_data):
    """Fills special values in the dataset based on annotated data.

    Args:
        data (np.ndarray): The input data array.
        annotated_data (np.ndarray): The annotated data array indicating special values.
    """
    for index, col in enumerate(annotated_data):
        # special is the column where we have specified the numbers representing "I don't know" and "not sure" ...
        # special becomes a list if there are multiple values separated by "&"
        special = [int(float(s)) for s in col[4].split("&")] 
        x_train[:, index] = np.where(np.isin(x_train[:, index], special), np.nan, x_train[:, index])
    return x_train

def preprocess_data(x_train_raw: np.ndarray, x_test_raw: np.ndarray, annotated_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Preprocesses the training and testing data by removing useless features.

    Args:
        x_train_raw (np.ndarray): The raw training data.
        x_test_raw (np.ndarray): The raw testing data.
        annotated_data (np.ndarray): The annotated data array indicating useful features.

    Returns:
        tuple[np.ndarray, np.ndarray]: The preprocessed training and testing data.
    """
    x_train = remove_useless(x_train_raw, annotated_data)
    x_test = remove_useless(x_test_raw, annotated_data)
    x_train = fill_specials(x_train, annotated_data)
    x_test = fill_specials(x_test, annotated_data)

    return x_train, x_test



def build_poly(x, degree, interactions=False):
    """
    Polynomial features up to 'degree'.
    - If x is (N,), returns (N, degree+1)  [bias + powers of the single feature]
    - If x is (N, D) and interactions=False, returns (N, 1 + D*degree)
      [bias + powers of each feature independently, no cross terms]
    - If x is (N, D) and interactions=True, returns all monomials up to total
      degree 'degree' (can be large): [1, x_i, x_i x_j, x_i^2, ...]
    """
    X = np.asarray(x)
    if X.ndim == 1:
        X = X.reshape(-1, 1)  # (N,1)

    N, D = X.shape

    if not interactions:
        # [1, X, X^2, ..., X^degree] without cross terms
        feats = [np.ones((N, 1))]
        for d in range(1, degree + 1):
            feats.append(X ** d)          
        return np.hstack(feats)           # (N, 1 + D*degree)

    # With interactions: all monomials up to total degree
    feats = [np.ones((N, 1))]
    # degree 1 terms are just the columns themselves
    feats.append(X)  # (N, D)

    # degrees >= 2: products of columns (with replacement)
    # e.g., deg=3, comb [0,0,1] -> X[:,0]^2 * X[:,1]
    for deg in range(2, degree + 1):
        for comb in combinations_with_replacement(range(D), deg):
            col = np.prod(X[:, comb], axis=1, keepdims=True)
            feats.append(col)
    return np.hstack(feats)


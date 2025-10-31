import numpy as np
from helpers import *
from itertools import combinations_with_replacement
from frequency_processing import *


def read_annotated_csv(path, delimiter=',', skip_header=0):
    """Reads our custom CSV file and returns the data as a NumPy array.
    This CSV file contains personal notes about each feature

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
            usecols=range(12),       
            filling_values=np.nan
        )[1:,1:12]

    return data

def preprocess_data(x_train_raw, y_train, x_test_raw, annotated_data, important_feat_only = False):
    """
    This function is responsible for the whole preprocessing of the datasets. It calls the relevant functions, namely : 
        - Remove features with excessive NaN values
        - Remove constant (non-informative) features based on the std of the feature
        - Handles categorical features with frequency-based encoding
        - Removes the useless features based on the artifically created csv
        - If Important feat only is True : we select only 22 features (for interaction) based on some paper, mentioned in the report
        - Cleans and encodes the data using clean_data : clean data description is available below
    Calls the relevant functions : 
            - removes the useless features based on the annotated csv files
            - cleans the dataset and encodes it
    Args : the raw datasets x_train_raw, y_train,x_test_raw and annotated data (csv file containing the informations about the features)

    Returns : x_train,y_train,x_test : the datasets contained in numpy arrays  
    """
    x_train_no_nan, x_test_no_nan, annotated_data = remove_nan(x_train_raw, x_test_raw,annotated_data)
    x_train_no_nan, x_test_no_nan, annotated_data = remove_constant(x_train_no_nan, x_test_no_nan,annotated_data)

    # dealing with the frequency issue
    x_train_no_nan = deal_with_frequencies(x_train_no_nan, annotated_data)
    x_test_no_nan = deal_with_frequencies(x_test_no_nan, annotated_data)

    #Remove useless features based on our annotated dataset
    x_train_filtered, mask = remove_useless(x_train_no_nan, annotated_data)
    x_test_filtered,mask = remove_useless(x_test_no_nan, annotated_data)
    annotated_data = annotated_data[mask,:]

    if important_feat_only is True : 
        x_train, categories_list = clean_data(x_train_filtered, annotated_data,important_feat_only=True)
        x_test,categories_list = clean_data(x_test_filtered, annotated_data,test = True,categories_list = categories_list,important_feat_only=True)
    else : 
        x_train, categories_list = clean_data(x_train_filtered, annotated_data)
        x_test,categories_list = clean_data(x_test_filtered, annotated_data,test = True,categories_list = categories_list)

      

    x_train, x_test = remove_constant_variance(x_train, x_test)
    x_train = np.hstack((np.ones((x_train.shape[0],1)), x_train))
    x_test = np.hstack((np.ones((x_test.shape[0],1)), x_test))
    return x_train, y_train, x_test

def remove_constant_variance(x_train, x_test):
    """Removes features with constant variance from the dataset.

    Args:
        x_train (np.ndarray): The training data array.
        x_test (np.ndarray): The testing data array.
    Returns:
        np.ndarray, np.ndarray: The training and testing data arrays with constant variance features removed.
        mask (np.ndarray): Boolean mask indicating which features were retained.
        """
    var = np.var(x_train, axis=0)
    mask = var != 0
    x_train = x_train[:, mask]
    x_test = x_test[:, mask]
    return x_train, x_test

def remove_nan(x_train, x_test, annotated_data):
    """
    Removes features (columns) with too many missing values (NaNs) from the dataset.

    This function calculates the proportion of NaN values in each feature of the 
    training set and removes any features whose NaN ratio exceeds 25%. 
    The same feature mask is then applied to the test set to ensure consistency

    Args : 
        x_train (np.ndarray) : the original training set
        x_test (np.ndarray) : the test set
    Returns : 
        x_train (np.ndarray) : the training set with the features with more than 25% NaNs removed
        x_test (np.ndarray) : the test set with the features with more than 25% NaNs (in train set) removed
        annotated_data (np.ndarray) : the artificial csv with the corresponding features removed (to keep track of indices!)
    """
    bool_nan = np.isnan(x_train)
    nan_ratio = np.mean(bool_nan, axis=0)
    mask = nan_ratio < 0.25
    x_train = x_train[:, mask]
    x_test = x_test[:, mask]
    annotated_data = annotated_data[mask,:]
    return x_train, x_test, annotated_data


def remove_constant(x_train, x_test,annotated_data):
    """Removes constant features from the dataset.

    Args:
        x_train (np.ndarray): The training data array.
        x_test (np.ndarray): The testing data array.
    Returns:
        np.ndarray, np.ndarray: The training and testing data arrays with constant features removed.
    """
    std_dev = np.std(x_train, axis=0)
    mask = std_dev != 0
    x_train = x_train[:, mask]
    x_test = x_test[:, mask]
    annotated_data = annotated_data[mask,:]
    return x_train, x_test, annotated_data


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
    return data[:, mask], mask


def clean_data(data, annotated_data, test = False, categories_list = None, important_feat_only = False) :
    """
    Cleans and processes the dataset by separating,transforming and merging back categorical and numerical features. 
    
    This function uses an annotated dataset ('data_annotated') which contains relevant informations concerning the features
    such as relevance (coded by binary yes/no), type (categorical,numerical), special formats, nb of categories for categorical features.

    This was done in order to : 
        - Remove a big chunk of irrelevant features
        - Handle special values (for example 'I dont know' or 'Prefer not to say' are sometimes coded as 9 and 7 but in fact they genuiely represent a NaN for our model)
        - Separate numerical and categorical features for One-Hot encoding
        - Normalize numerical values 
        - Merge back the categorical and numerical datasets once they've been cleaned 
    Args: 
        data (np.ndarray) : Original dataset containing all the features
        data_annotated (np.ndarray) : Annotated dataset containing all the relevant informations about the features
    Returns : 
        data_clean (np.ndarray) : Fully processed dataset with encoded categorical variables and normalized numerical features
    """
    #Remove useless features : 
    if important_feat_only is True : 
        #Identify categorical and numerical features : 
        idx_categorical = np.where(((annotated_data[:,1] != '0') | (annotated_data[:,2]!= '0')) & (annotated_data[:,7] == '1'))[0]
        idx_numerical = np.where((annotated_data[:,3]!= '0')& (annotated_data[:,7]== '1'))[0]     
                            
    else : 
        #Identify categorical and numerical features : 
        idx_categorical = np.where((annotated_data[:,1] != '0') | (annotated_data[:,2]!= '0'))[0]
        idx_numerical = np.where(annotated_data[:,3]!= '0')[0]
    
    #Split Data into separate categories (numerical and categorical)
    data_categorical = data[:,idx_categorical] 
    data_numerical = data[:,idx_numerical] 
    data_annotated_categorical = annotated_data[idx_categorical,:] 
    data_annotated_numerical = annotated_data[idx_numerical,:]

    #Handle special values 
    data_numerical =  deal_with_specials(data_annotated_numerical, data_numerical)
    data_categorical = deal_with_specials(data_annotated_categorical, data_categorical)

    #Replace the remaining NaN's (when % of Nans of the feature is <25%) of the numerical features by the median value of the feature
    col_median = np.nanmedian(data_numerical, axis = 0)
    mask_nan = np.where(np.isnan(data_numerical))
    data_numerical[mask_nan] = np.take(col_median, mask_nan[1])
    numerical_normalized = normalize_data(data_numerical)
    

    #One-hot encode categorical features
    if test is True : 
        data_categorical, categories_list = one_hot_encode(data_categorical, data_annotated_categorical,categories_list)
    else : 
        data_categorical, categories_list = one_hot_encode(data_categorical, data_annotated_categorical)

    #Merge categorical & numerical
    data_clean = np.hstack([data_categorical, numerical_normalized])
    
    return data_clean,categories_list

def deal_with_specials(annotated_data, data):
    """
    Replaces special or invalid feature values (e.g., "don't know", "prefer not to say") with NaN.

    This function uses the annotated dataset (`data_annotated`) to identify special numeric codes 
    for each feature (stored in the 5th column). These codes are replaced by NaN in the corresponding 
    columns of the main dataset.

    Args:
        data_annotated (np.ndarray): Annotated data for each feature, where the 5th column lists 
                                  special values separated by '&'
        data (np.ndarray): Original dataset containing feature values

    Returns:
        data (np.ndarray): Modified dataset with all special values replaced by NaN.
    """

    for index in range(len(annotated_data)): 
        #Get the special values (dont know & prefer not to say) which are stored in the 5th column of 'data_annotated'
        special_vals = annotated_data[index,4] 

       #Split the special values when there are more than one 
        special_val = special_vals.split('&') 

        #Iterate through the special values
        for val in special_val :
            val = float(val)
            
            #Gets the indices where the feature = special value
            mask = data[:,index] == val 

            #Replace the special value by a NaN
            data[mask,index] = np.nan 
    return data


def normalize_data(data):
    """
    Normalizes numerical features to 0-1 range using min-max scaling
    Each feature (column) is scaled independently according to : 
        x_norm = (x-min(x))/(max(x)- min(x))
    
    Args : 
        - data (np.ndarray) : numerical dataset to be normalized
    Returns : 
        - data (np.ndarray) : numerical normalized dataset with values lying between 0 and 1
    """
    data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    return data


def one_hot_encode(data,annotated_data, categories = None) : 
    """
    One-hot encodes categorical features using the annotated dataset ('data_annotated')

    This function transforms each categorical feature into a one-hot encoded representation. 
    The number of category for each feature is known thanks to the annotated dataset. 
    Missing values (NaN) are treated as a separate category during encoding, but the corresponding
    column is removed afterward to avoid multicollinearity (dummy variable trap). 

    To ensure consistent encoding between training and test sets, the function can either : 
        - Create categories by itself (default, when 'categories = None') or 
        - Use predefined categories (the ones created for the training set)

    Args : 
        - data (np.ndarray) : Categorical data to be one-hot encoded
        - data_annotated (np.ndarray) : Annotated data for each feature,
          where the 7th column contains the nb of categories of each feature 
        - categories (np.ndarray) : Predefined categories for each feature to avoid any mismatch in the dimensions 
        of train and test sets. If None, categories are inferred from the data 
    
    Returns : 
        - one_hot (np.ndarray) : One-hot encoded dataset
        - categories_list list[np.ndarray] : list of np arrays containing the categories for each feature

    """
    categories_list = []
    one_hot_features = []
    n_feat = len(annotated_data)
    for index in range(n_feat): 

        #Extract nb of categories of each feature which is stored in data_annotated
        n = float(annotated_data[index,6]) 

        #Select all samples from the feature we currently are one-hot encoding
        feature = data[:,index] 
        #Create an array with all possibles for this feature
        unique_values = np.arange(1,n+1)
        if categories is None : 
             #Deal with NaN : also create a one-hot encoded column for NaN values
            if np.isnan(feature).any():
                #If there is NaN value : Add a category
                cat_values = np.concatenate([unique_values, [np.nan]])
            else:
                cat_values = unique_values 
        else :
            cat_values = categories[index]
             
        categories_list.append(cat_values)
       #Create a matrix of zeros with shape (nb of samples, nb of categories of the feature) 
        one_hot_matrix = np.zeros((data.shape[0], len(cat_values)))
        
        #Go through each sample of the feature and fill the one_hot_matrix with the right row and column
        for i,value in enumerate(feature): 
                if np.isnan(value): 
                        cat_idx = np.where(np.isnan(cat_values))[0] 
                else:
                        cat_idx = np.where(cat_values == value)[0] 
                one_hot_matrix[i, cat_idx] = 1
        
        #Remove the last column of each matrix (corresponding to category NaN) to avoid multicolinearity aka Dummy Variable trap
        one_hot_matrix = one_hot_matrix[:,:-1] 
        one_hot_features.append(one_hot_matrix)
       
    
    onehot = np.hstack(one_hot_features)
    
    return onehot, categories_list

def balance_data(y, x):
    """
    Balances a binary dataset by randomly undersampling the majority class

    This function ensures that both classes (e.g., positive and negative samples)
    have an equal number of observations by randomly removing samples from the 
    majority class. It is particularly useful when dealing with imbalanced datasets 
    where one class is significantly overrepresented and avoids the models to be
    biased toward the majority class

    Args : 
        - y (np.ndarray) : target labels
        - x (np.ndarray) : feature matrix corresponding to the y samples
    
    Returns : 
        - x balanced (np.ndarray) : feature matrix with balanced dataset
        - y balanced (np.ndarray) : target labels after balancing the dataset
    """
    #Set Seed to have reproducible results when using random choices
    rng = np.random.default_rng(seed=42) 
    #Get the indxes of the majority & minority classes
    idx_majority = np.where(y==0)[0] 
    idx_minority = np.where(y==1)[0] 

    #Randomly takes rows from the majority class and removes them so that we have the same number of samples in both classes
    idx_majority_sampled = rng.choice(idx_majority, size=len(idx_minority), replace=False)

    #Combine back the indices
    idx_balanced = np.concatenate([idx_majority_sampled, idx_minority])

    #Reshuffle the indices 
    rng.shuffle(idx_balanced)
    return x[idx_balanced], y[idx_balanced]





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


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing. If ratio times the number of samples is not round
    you can use np.floor. Also check the documentation for np.random.permutation,
    it could be useful.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.

    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.

    """
    np.random.seed(seed)
    idx = np.random.permutation(x.shape[0])
    split_idx = int(ratio*x.shape[0])
    tr_idx = idx[:split_idx]
    te_idx = idx[split_idx:]
    return x[tr_idx],x[te_idx],y[tr_idx],y[te_idx]
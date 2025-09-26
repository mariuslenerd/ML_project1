import numpy as np

def find_categorical(x, threshold = 20) : 
    """
    This function finds and returns the 2 lists of indexes : 1 for which the feature is categorical 
    and 1 for which the feature is numerical. It will be later used for one-hot-encode the categorical features

    The function looks at the number of unique values for each feature. 
    If the number of unique values is > 20 the feature is flagged as categorical
    If the number of unique values is < 20 the feature is flagged as numerical
    Args : 
        x - the features dataset
        threshold - the threshold for which nb of unique values > threshold : considered numerical and 
        nb of unique values < threshold : considered categorical
    
    Returns : 
        one list of indexes that are categorical features, one list of indexes that are numerical features
    
    """
    num_features = x.shape[1]
    numerical_idx = []
    categorical_idx = []
    for feature in range(num_features) : 
        full_column = x[:,feature]
        nb_unique_vals = len(np.unique(full_column))
        if nb_unique_vals < threshold : 
            categorical_idx.append(feature)
        else : 
            numerical_idx.append(feature)
    return numerical_idx, categorical_idx

def one_hot_encode(x) : 

    numerical_idx, categorical_idx = find_categorical(x)
    x_numerical = x[:, numerical_idx]
    x_categorical = x[:,categorical_idx]
    
   


        """
        x_train_one_hot_encode = x_train_raw[:,1]
        unique_vals = np.unique(x_train_one_hot_encode)
        print(np.unique(x_train_one_hot_encode)) #my array has value from 1 to 12 : 12 categories -> want to create Nx12 matrix filled up with 0 except for when sample N belong to category i : filled with 1 
        #create maping : 
        mapping = {float(val) : i for i,val in enumerate(unique_vals)}
        display(mapping)
        x_mapped =  np.array([mapping[v] for v in x_train_one_hot_encode])
        x_encoded = np.eye(len(unique_vals))[x_mapped]

        display(x_encoded)
        #x_train_encoded = np.eye(len(np.unique(x_train_one_hot_encode)))[x_train_one_hot_encode]"""
    return x
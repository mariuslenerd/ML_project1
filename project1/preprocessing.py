def find_categorical(data) : 
    """
    This function finds and returns the 2 lists of indexes : 1 for which the feature is categorical 
    and 1 for which the feature is numerical. It will be later used for one-hot-encode the categorical features

    The function looks at the number of unique values for each feature. 
    If the number of unique values is > 20 the feature is flagged as categorical
    If the number of unique values is < 20 the feature is flagged as numerical
    Args : data - the features dataset
    Returns : one list of indexes that are categorical features, one list of indexes that are numerical features
    
    """
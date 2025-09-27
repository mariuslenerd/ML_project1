import numpy as np
from helpers import *

def remove_useless(data: np.ndarray, annotated_data: np.ndarray) -> np.ndarray:
    """Removes useless features from the dataset based on annotated data.

    Args:
        data (np.ndarray): The input data array.
        annotated_data (np.ndarray): The annotated data array indicating useful features.

    Returns:
        np.ndarray: The data array with useless features removed.
    """
    mask = annotated_data[:, 1] != 0               
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
        dtype=float,            # or None
        autostrip=True,
        comments=None,
        invalid_raise=False,    # don't error on inconsistent rows
        usecols=range(7),       # force 7 columns
        filling_values=np.nan
    )[1:,1:6]
    return data  

def fill_empty_with_nan(data: np.ndarray) -> np.ndarray:
    """Fills empty values in the dataset with NaN.

    Args:
        data (np.ndarray): The input data array.

    Returns:
        np.ndarray: The data array with empty values filled with NaN.
    """
    data_filled = np.where(data == '', np.nan, data)
    return data_filled

def fill_specials(data, annotated_data):
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




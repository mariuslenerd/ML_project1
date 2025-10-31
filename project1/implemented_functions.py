# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np


def batch_iter(y,tx,batch_size, num_batches = 1, shuffle = True) : 
    """
        Generate a minibatch iterator for a dataset.
        Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
        Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
        Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index], np.arange(start_index, end_index)


def make_class_weights(y):
    """
    Create sample weights to balance classes in binary classification.
    Args:
        y (np.array): binary labels (0 or 1), shape (N,)
    Returns:
        np.array: sample weights, shape (N,)
    """
    N = y.shape[0]
    N1 = np.sum(y == 1)
    N0 = np.sum(y == 0)

    w1 = N / (2 * N1) if N1 > 0 else 0.0
    w0 = N / (2 * N0) if N0 > 0 else 0.0

    sample_weights = np.where(y == 1, w1, w0)
    return sample_weights

def compute_weighted_loss(y, x, w, weights):
    """
    Compute weighted mean squared error loss.
    Args:
        y (np.array): true labels, shape (N,)
        x (np.array): input data, shape (N,D)
        w (np.array): model parameters, shape (D,)
        weights (np.array): sample weights, shape (N,)
    Returns:
        float: weighted MSE loss
    """
    residuals = y - x @ w
    weighted_sq_errors = weights * (residuals ** 2)
    return 0.5*np.sum(weighted_sq_errors) / np.sum(weights)


def compute_loss(y,tx,w) : 
    """
    Calculate the loss using Mean Squared Error

    Args : 
        y : numpy array of shape = (N,)
        tx : numpy array of shape = (N,D)
        w: numpy array of shape = (D,). The vector of model parameters

    Returns : 
        the value of the square loss (a scalar) corresponding to input parameters w
    """
    N = y.shape[0]
    e = y - tx @ w
    MSE_loss = 1 / (2*N) * (e.T @ e)
    return MSE_loss

def compute_rmse(y,tx,w) : 
    """
    Computes the Root Mean Square Error (RMSE) at w
    Args : 
        y : numpy array of shape = (N,)
        tx : numpy array of shape = (N,D)
        w: numpy array of shape = (D,). The vector of model parameters
    Returns :
        the value of the RMSE (a scalar) corresponding to input parameters w
    """
    return np.sqrt(2*compute_loss(y,tx,w))

def compute_mse_gradient(y,tx,w, weights = None) : 
    """
    Computes the gradient at w using the mse loss 
    
    Args : 
        y : numpy array of shape = (N,)
        tx : numpy array of shape = (N,D)
        w: numpy array of shape = (D,). The vector of model parameters

    Returns : 
        Numpy array of shape (D,) containing the gradient of the loss at w
    """
    if weights is None:
        weights = make_class_weights(y)
    n = len(y)
    e = y- tx@w
    return -(tx.T@(weights*e))/np.sum(weights)

# ----------- GRADIENT DESCENT -----------

def mean_squared_error_gd(y,tx,initial_w, max_iters,gamma) : 
    """
    Performs gradient descent with MSE loss and returns the last computed parameters 
    
    Args : 
        y : numpy array of shape = (N,)
        tx : numpy array of shape = (N,D)
        initial_w: numpy array of shape = (D,). The vector of initial model parameters
        max_iters : int. The number of iteration of the gradient descent algorithm
        gamma : float. The learning rate, i.e the stepsize
    
    Returns : 
        A numpy array of shape (D,), the same as initial_w. containing the last vector of parameters and a float corresponding to the loss value of the model with those last parameters  
    """

    w = initial_w
    loss = compute_weighted_loss(y,tx,w,weights=make_class_weights(y))
    for n in range(max_iters) : 
        w = w - gamma*compute_mse_gradient(y,tx,w)

        loss = compute_weighted_loss(y,tx,w,weights=make_class_weights(y))

    return w,loss

# ----------- STOCHASTIC GRADIENT DESCENT -----------

def mean_squared_error_sgd(y,tx,initial_w, max_iters, gamma) : 
    """
    Performs stochastic gradient descent (note that batch_size = 1, specified in the project description) using mse loss and returns the last computed parameters along with its associated loss

    Args : 
        y : numpy array of shape = (N,)
        tx : numpy array of shape = (N,D)
        initial_w: numpy array of shape = (D,). The vector of initial model parameters
        max_iters : int. The number of iteration of the gradient descent algorithm
        gamma : float. The learning rate, i.e the stepsize

    Returns : 
        A numpy array of shape (D,), the same as initial_w. containing the last vector of parameters and a float corresponding to the loss value of the model with those last parameters
    """
    batch_size = 32

    w = initial_w
    weights = make_class_weights(y)

    for mini_batch_y, mini_batch_tx, idx in batch_iter(y,tx,batch_size,max_iters): 
        batch_weights = weights[idx]
        grad = compute_mse_gradient(mini_batch_y,mini_batch_tx,w,weights=batch_weights)
        w = w-gamma*grad
    loss = compute_weighted_loss(y,tx,w,weights)

    return w, loss

# ----------- LEAST SQUARES -----------


def class_weighted_least_squares(y, tx):
    """
    Solve least squares but force each class to matter equally.

    Args:
        y  : shape (N,), values in {0,1}
        tx : shape (N,D)

    Returns:
        w     : shape (D,)
        loss  : scalar (weighted MSE)
    """
    sample_weights = make_class_weights(y)       

    sqrt_w = np.sqrt(sample_weights)[:, None]    
    X_tilde = sqrt_w * tx                        
    y_tilde = np.sqrt(sample_weights) * y        

    w = np.linalg.lstsq(X_tilde, y_tilde, rcond=None)[0]

    loss = compute_weighted_loss(y, tx, w, sample_weights)
    return w, loss

def least_squares(y,tx) : 
    """
    Performs the ordinary least squares regression algorithm and returns the best parameters, along with its associated loss

    Args : 
        y : numpy array of shape = (N,)
        tx : numpy array of shape = (N,D)
    
    Returns : 
        w : optimal weights, numpy array of shape (D,) where D is the nb of features 
        loss : scalar, mean squared error loss
    """
    w = np.linalg.lstsq(tx,y)[0]
    loss = compute_loss(y,tx,w)

    return w,loss

# ----------- RIDGE REGRESSION -----------

def compute_ridge_loss(y, tx, w, lambda_):
    """
    Compute the ridge loss.

    Args:
        y (np.array): true labels, shape (N,)
        tx (np.array): input data, shape (N,D)
        w (np.array): model parameters, shape (D,)
        lambda_ (float): regularization strength

    Returns:
        float: ridge loss§
    """

    sample_weights = make_class_weights(y) 

    N = y.shape[0]
    e = y - tx @ w
    ridge_loss = 0.5*np.mean((sample_weights*(e**2))) + 0.5 * lambda_ * w.T @ w

    return ridge_loss

def ridge_regression(y, tx, lambda_):
    """
    Perform ridge regression.

    Args:
        y (np.array): true labels, shape (N,)
        tx (np.array): input data, shape (N,D)
        lambda_ (float): regularization strength

    Returns:
        w (np.array): model parameters, shape (D,)
        loss (float): ridge loss
    """

    N, D = tx.shape
    weights = make_class_weights(y)
    A = tx.T @ (weights[:, np.newaxis] * tx) + (N * lambda_) * np.eye(D)
    b = tx.T @ (weights * y)
                                    
    w = np.linalg.solve(A, b)  
    loss = compute_ridge_loss(y, tx, w, lambda_)
    return w, loss

# ----------- LOGISTIC REGRESSION -----------

def sigmoid(z):
    """
    Compute the sigmoid function.
    Args:
        z (np.array): input array
    Returns:
        np.array: sigmoid of input array
    """
    return 1 / ( 1 + np.exp(-z))

def compute_logistic_loss(y, tx, w):
    """
    Compute the logistic loss.
    Args:
        y (np.array): true labels, shape (N,)
        tx (np.array): input data, shape (N,D)
        w (np.array): model parameters, shape (D,)
    Returns:
        float: logistic loss
    """
    w1 = tx.shape[0]/(2*np.sum(y == 1))
    w0 = tx.shape[0]/(2*np.sum(y == 0))
    prediction = sigmoid(tx @ w)
    loss = -np.mean( w1 * y * np.log(prediction + 1e-15) + w0 * (1-y) * np.log(1 - prediction + 1e-15) )
    return loss

def compute_logistic_gradient(y, tx, w):
    """
    Compute the gradient of the logistic loss.
    Args:
        y (np.array): true labels, shape (N,)
        tx (np.array): input data, shape (N,D)
        w (np.array): model parameters, shape (D,)
    Returns:
        np.array: gradient of the logistic loss, shape (D,)
    """
    N = len(y)
    prediction = sigmoid(tx @ w)
    weights = make_class_weights(y)
    grad = (tx.T @ (weights * (prediction - y))) / N
    return grad

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Perform logistic regression using gradient descent.

    Args:
        y (np.array): true labels, shape (N,)
        tx (np.array): input data, shape (N,D)
        initial_w (np.array): initial model parameters, shape (D,)
        max_iters (int): maximum number of iterations
        gamma (float): learning rate

    Returns:
        np.array: final model parameters, shape (D,)
        float: final loss
    """
    w = initial_w
    for _ in range(max_iters):
        gradient = compute_logistic_gradient(y, tx, w)
        w = w - gamma * gradient
    loss = compute_logistic_loss(y, tx, w)
    return w, loss

# ---------- REGULARIZED LOGISTIC REGRESSION ----------

def compute_reg_logistic_loss(y, tx, w, lambda_):
    """
    Compute the regularized logistic loss.
    Args:
        y (np.array): true labels, shape (N,)
        tx (np.array): input data, shape (N,D)
        w (np.array): model parameters, shape (D,)
        lambda_ (float): regularization strength
    Returns:
        float: regularized logistic loss
    """
    prediction = sigmoid(tx @ w)
    w1 = len(y) / (2 * np.sum(y == 1))
    w0 = len(y) / (2 * np.sum(y == 0))
    cross_entropy_loss = -np.mean(w1 * y * np.log(prediction + 1e-15) + w0 * (1-y) * np.log(1 - prediction + 1e-15))
    regularization = 0.5 * lambda_ * np.sum(w[1:] ** 2)
    return cross_entropy_loss + regularization

def compute_reg_logistic_gradient(y, tx, w, lambda_):
    """
    Compute the gradient of the regularized logistic loss.
    Args:
        y (np.array): true labels, shape (N,)
        tx (np.array): input data, shape (N,D)
        w (np.array): model parameters, shape (D,)
        lambda_ (float): regularization strength
    Returns:
        np.array: gradient of the regularized logistic loss, shape (D,)
    """
    N = y.shape[0]
    prediction = sigmoid(tx @ w)
    weights = make_class_weights(y)
    gradient = 1/N  * tx.T @ (weights *(prediction - y))
    gradient[1:] += lambda_ * w[1:]
    return gradient

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Perform regularized logistic regression using gradient descent.
    Args:
        y (np.array): true labels, shape (N,)
        tx (np.array): input data, shape (N,D)
        lambda_ (float): regularization strength
        initial_w (np.array): initial model parameters, shape (D,)
        max_iters (int): maximum number of iterations
        gamma (float): learning rate
    Returns:
        np.array: final model parameters, shape (D,)
        float: final loss
    """
    w = initial_w
    for _ in range(max_iters):
        gradient = compute_reg_logistic_gradient(y, tx, w, lambda_)
        w = w - gamma * gradient
    loss = compute_reg_logistic_loss(y, tx, w, lambda_)
    return w, loss


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
        yield y[start_index:end_index], tx[start_index:end_index]

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
    return np.sqrt(2*compute_loss(y,tx,w))

def compute_mse_gradient(y,tx,w) : 
    """
    Computes the gradient at w using the mse loss 
    
    Args : 
        y : numpy array of shape = (N,)
        tx : numpy array of shape = (N,D)
        w: numpy array of shape = (D,). The vector of model parameters

    Returns : 
        Numpy array of shape (D,) containing the gradient of the loss at w
    """
    n = len(y)
    
    #error vector 
    e = y- tx@w
    return -(tx.T@e)/n

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
    loss = compute_loss(y,tx,w)

    for n in range(max_iters) : 
        w = w - gamma*compute_mse_gradient(y,tx,w)

        loss = compute_loss(y,tx,w)

    return w,loss

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
    batch_size = 1

    w = initial_w
    loss = compute_loss(y,tx,w)

    for mini_batch_y, mini_batch_tx in batch_iter(y,tx,batch_size,max_iters): 
        grad = compute_mse_gradient(mini_batch_y,mini_batch_tx,w)
        w = w-gamma*grad
        loss = compute_loss(y,tx,w)

    return w,loss

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

def compute_ridge_loss(y, tx, w, lambda_):
    N = y.shape[0]
    e = y - tx @ w
    ridge_loss = 1 / (2*N) * (e.T @ e) + 0.5 * lambda_ * w.T @ w
    return ridge_loss

def ridge_regression(y, tx, lambda_):
    N, D = tx.shape
    A = tx.T @ tx + (N * lambda_) * np.eye(D)      
    b = tx.T @ y                                   
    w = np.linalg.solve(A, b)  
    loss = compute_ridge_loss(y, tx, w, lambda_)
    return w, loss
# ----------- LOGISTIC REGRESSION -----------
def sigmoid(z):
    return 1 / ( 1 + np.exp(-z))

def compute_logistic_loss(y, tx, w):
    prediction = sigmoid(tx @ w)
    loss = -np.mean(y * np.log(prediction + 1e-15) + (1-y) * np.log(1 - prediction + 1e-15))
    return loss

def compute_logistic_gradient(y, tx, w):
    prediction = sigmoid(tx @ w)
    return 1/tx.shape[0] * tx.T @ (prediction - y)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for _ in range(max_iters):
        gradient = compute_logistic_gradient(y, tx, w)
        w = w - gamma * gradient
    loss = compute_logistic_loss(y, tx, w)
    return w, loss

# ---------- REGULARIZED LOGISTIC REGRESSION ----------

def compute_reg_logistic_loss(y, tx, w, lambda_):
    prediction = sigmoid(tx @ w)
    cross_entropy_loss = -np.mean(y * np.log(prediction + 1e-15) + (1-y) * np.log(1 - prediction + 1e-15))
    # TODO: Check if we should also add the regularization term on the bias ? If so remove [1:]
    regularization = lambda_ * 0.5 * w.T[1:] @ w[1:]
    return cross_entropy_loss + regularization

def compute_reg_logistic_gradient(y, tx, w, lambda_):
    N = y.shape[0]
    prediction = sigmoid(tx @ w)
    # TODO: Check if we should also add the regularization term on the bias ?  if so remove [1:]
    gradient = 1/N  * tx.T @ (prediction - y)
    gradient[1:] += lambda_ * w[1:]
    return gradient

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    for _ in range(max_iters):
        gradient = compute_reg_logistic_gradient(y, tx, w, lambda_)
        w = w - gamma * gradient
    loss = compute_reg_logistic_loss(y, tx, w, lambda_)
    return w, loss



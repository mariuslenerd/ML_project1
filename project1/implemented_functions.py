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

    n = len(y)
    return (1/2*n)*np.sum((y-tx@w)**2)

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

def gradient_descent(y,tx,initial_w, max_iters,gamma) : 
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

def stochastic_gradient_descent(y,tx,initial_w, max_iters, gamma) : 
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

    for n_iter,(mini_batch_y, mini_batch_tx) in enumerate(batch_iter(y,tx,batch_size,max_iters)) : 
        grad = compute_mse_gradient(mini_batch_y,mini_batch_tx,w)
        w = w-gamma*grad
        loss = compute_loss(y,tx,w)

    return w,loss

def ordinary_least_squares(y,tx) : 
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



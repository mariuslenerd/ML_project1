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
    batch_size = 8

    w = initial_w
    loss = compute_loss(y,tx,w)
    weights = make_class_weights(y)
    for mini_batch_y, mini_batch_tx, idx in batch_iter(y,tx,batch_size,max_iters): 
        batch_weights = weights[idx]
        grad = compute_mse_gradient(mini_batch_y,mini_batch_tx,w,weights=batch_weights)
        w = w-gamma*grad
    loss = compute_weighted_loss(y,tx,w,weights)

    return w, loss

def make_class_weights(y):
    """
    y: shape (N,), binary {0,1}
    returns sample_weights: shape (N,)
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
    attention! weights is not the weights of the model but the weights for class balancing
    """
    residuals = y - x @ w
    weighted_sq_errors = weights * (residuals ** 2)
    return 0.5*np.sum(weighted_sq_errors) / np.sum(weights)

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

def compute_ridge_loss(y, tx, w, lambda_):
    # with weighted loss 
    w1 = tx.shape[0]/(2*np.sum(y == 1))
    w0 = tx.shape[0]/(2*np.sum(y == 0))
    weights = np.where(y == 1, w1, w0)
    N = y.shape[0]
    e = y - tx @ w
    ridge_loss = 0.5*np.mean((weights*(e**2))) + 0.5 * lambda_ * w.T @ w
    # without weighted loss
    #N = y.shape[0]
    #e = y - tx @ w
    #ridge_loss = 1 / (2*N) * (e.T @ e) + 0.5 * lambda_ * w.T @ w
    return ridge_loss

def ridge_regression(y, tx, lambda_):
    N, D = tx.shape
    # implement class weights test:
    w1 = tx.shape[0]/(2*np.sum(y == 1))
    w0 = tx.shape[0]/(2*np.sum(y == 0))
    weights = np.where(y == 1, w1, w0)
    # with weights 
    A = tx.T @ (weights[:, np.newaxis] * tx) + (N * lambda_) * np.eye(D)
    b = tx.T @ (weights * y)
    # without weights
    #A = tx.T @ tx + (N * lambda_) * np.eye(D)      
    #b = tx.T @ y                                   
    w = np.linalg.solve(A, b)  
    loss = compute_ridge_loss(y, tx, w, lambda_)
    return w, loss
# ----------- LOGISTIC REGRESSION -----------
def sigmoid(z):
    return 1 / ( 1 + np.exp(-z))

def compute_logistic_loss(y, tx, w):
    w1 = tx.shape[0]/(2*np.sum(y == 1))
    w0 = tx.shape[0]/(2*np.sum(y == 0))
    prediction = sigmoid(tx @ w)
    loss = -np.mean( w1 * y * np.log(prediction + 1e-15) + w0 * (1-y) * np.log(1 - prediction + 1e-15) )
    return loss

def compute_logistic_gradient(y, tx, w):
    N = len(y)
    prediction = sigmoid(tx @ w)
    w1 = N / (2 * np.sum(y == 1))
    w0 = N / (2 * np.sum(y == 0))
    weights = np.where(y == 1, w1, w0)
    grad = (tx.T @ (weights * (prediction - y))) / N
    return grad

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
    w1 = len(y) / (2 * np.sum(y == 1))
    w0 = len(y) / (2 * np.sum(y == 0))
    cross_entropy_loss = -np.mean(w1 * y * np.log(prediction + 1e-15) + w0 * (1-y) * np.log(1 - prediction + 1e-15))
    # TODO: Check if we should also add the regularization term on the bias ? If so remove [1:]
    regularization = 0.5 * lambda_ * np.sum(w[1:] ** 2)
    return cross_entropy_loss + regularization

def compute_reg_logistic_gradient(y, tx, w, lambda_):
    N = y.shape[0]
    prediction = sigmoid(tx @ w)
    w1 = N / (2 * np.sum(y == 1))
    w0 = N / (2 * np.sum(y == 0))
    weights = np.where(y == 1, w1, w0)
    # TODO: Check if we should also add the regularization term on the bias ?  if so remove [1:]
    gradient = 1/N  * tx.T @ (weights *(prediction - y))
    gradient[1:] += lambda_ * w[1:]
    return gradient

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w
    for _ in range(max_iters):
        gradient = compute_reg_logistic_gradient(y, tx, w, lambda_)
        w = w - gamma * gradient
    loss = compute_reg_logistic_loss(y, tx, w, lambda_)
    return w, loss


# ---------------- LASSO ----------------

def sigmoid(z):
    # stable sigmoid
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))

def soft_threshold(v, thr):
    # elementwise soft-thresholding
    return np.sign(v) * np.maximum(np.abs(v) - thr, 0.0)

# ===== L1 (lasso) loss =====
def compute_reg_logistic_loss_l1(y, tx, w, lambda_):
    """
    Logistic loss + L1 penalty (no penalty on bias w[0]).
    """
    p = sigmoid(tx @ w)
    w1 = 1/np.sum(y == 1)
    w0 = 1/np.sum(y == 0)
    # cross-entropy
    ce = -np.mean(w1 * y * np.log(p + 1e-15) + w0 * (1 - y) * np.log(1 - p + 1e-15))
    # L1 penalty (note: no 0.5 factor for L1)
    l1 = lambda_ * np.sum(np.abs(w[1:]))
    return ce + l1

def compute_logistic_grad_no_reg(y, tx, w):
    """
    Gradient of logistic loss ONLY (no regularization).
    """
    N = y.shape[0]
    p = sigmoid(tx @ w)
    return (tx.T @ (p - y)) / N

# ===== Proximal Gradient Descent (ISTA) for L1-regularized logistic regression =====
def reg_logistic_lasso(y, tx, lambda_, initial_w, max_iters, gamma, tol=1e-8):
    """
    Proximal gradient (soft-thresholding) solver for L1-regularized logistic regression.
    - No penalty on bias w[0].
    - gamma is the step size (can use backtracking; here it's fixed).
    """
    w = initial_w.astype(float).copy()
    for _ in range(max_iters):
        grad = compute_logistic_grad_no_reg(y, tx, w)
        w_old = w.copy()
        # gradient step
        w = w - gamma * grad
        # proximal (soft-threshold) on non-bias weights only
        w[1:] = soft_threshold(w[1:], gamma * lambda_)
        # stopping rule
        if np.linalg.norm(w - w_old, ord=np.inf) <= tol * max(1.0, np.linalg.norm(w_old, ord=np.inf)):
            break
    loss = compute_reg_logistic_loss_l1(y, tx, w, lambda_)
    return w, loss

# ===== OPTIONAL: Subgradient version (simpler but slower) =====
def compute_reg_logistic_subgradient(y, tx, w, lambda_):
    """
    Subgradient of logistic loss + L1.
    Uses 0 as subgradient at exactly 0 (valid choice).
    """
    grad = compute_logistic_grad_no_reg(y, tx, w)
    sg = np.sign(w)
    sg[np.abs(w) < 1e-12] = 0.0
    sg[0] = 0.0  # do not regularize bias
    grad += lambda_ * sg
    return grad

def reg_logistic_lasso_subgradient(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w.astype(float).copy()
    for _ in range(max_iters):
        grad = compute_reg_logistic_subgradient(y, tx, w, lambda_)
        w = w - gamma * grad
    loss = compute_reg_logistic_loss_l1(y, tx, w, lambda_)
    return w, loss

def compute_f1_score(y_true, y_pred):
    """
    Compute the F1 score given true and predicted binary labels.

    Args:
        y_true: numpy array of shape (N,), true binary labels (0 or 1)
        y_pred: numpy array of shape (N,), predicted binary labels (0 or 1)
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    f1 = 2*true_positives / (2*true_positives + false_positives + false_negatives)
    return f1

def compute_accuracy(y_test, x_test, w, method, threshold=0.5, mode=None, detailed=True):
    
    if method in ["logistic", "Regularized Logistic", "reg_lasso_logistic"]:
        y_pred = sigmoid(x_test@w)
    else:
        y_pred = x_test@w
    y_pred[y_pred <= threshold] = 0
    y_pred[y_pred > threshold] = 1
    if mode == 'submission':
        # transform all 0 predictions to -1 for submission format
        y_pred[y_pred == 0] = -1
        return 0, y_pred
    computed_accuracy = np.sum(y_pred == y_test)/len(y_test)
    if detailed:
        print(f"Accuracy of {method} is {computed_accuracy*100}%")
    return computed_accuracy, y_pred

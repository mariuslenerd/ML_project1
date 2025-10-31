import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_nans_hist(data): 
    """
    This function plots the distribution of missing values (NaNs) across columns 
    in a dataset by counting the number of NaN values in each column of the input data 
    and plots a histogram showing how many columns have a given number of missing values. 

    Args :
        data(np.ndarray) : Shape (N,D) where N = nb of samples and D = nb of features
            The dataset to analyze where columns represent features.

    Returns
        None
    """
    nan_counts = np.isnan(data).sum(axis=0)
    plt.figure(figsize=(8,6))
    plt.hist(nan_counts, bins=50,color = 'red', edgecolor='black')
    plt.xlabel("Number of NaNs in a column")
    plt.ylabel("Number of columns")
    plt.title("Distribution of NaNs per column")
    plt.show()

def plot_data_imbalance(data) : 
    """
    This function plots the class distribution of the binary target variable (y_train) 
    in order to better visualize data imbalance.

    We count the number of samples for each class in the target data 
    (e.g., presence or absence of a heart attack) and generate a bar plot 
    showing the relative frequencies.

        Args : 
            data (np.ndarray) : 1D array of binary target variables, 
                1 represents heart attack while -1 represent no heart attack
        Returns : 
            None
    """
    count_ill = (data== 1).sum()
    count_not_ill = (data== -1).sum()
    counts = np.array(count_ill,count_not_ill)
    categories = np.array(['Heart attack', ' No heart attack'])

    counts = np.array([count_ill, count_not_ill]) 
    sort_indices = np.argsort(counts)[::-1]
    sorted_categories = categories[sort_indices]
    sorted_counts = counts[sort_indices]
    plt.figure(figsize=(8,6))
    plt.title('Heart attack vs no heart attack')
    plt.ylabel('nb of sample')
    plt.bar(sorted_categories, sorted_counts, color=['red', 'orange'])

def plot_cv_results(curves, method, degree, show_train=True, log_x=True):
    """
    This function plots the cross-validation results for a single method
    at a specific polynomial expansion degree.

    The type of visualization depends on the hyperparameters used by the method : 
    - if neither lambda nor gamma are used : simply prints the train and test losses
    - if only lambda is used : plots the test/train loss vs lambda
    - if  only gamma is used : plots the test/train loss vs gamma
    - if method uses both lambda and gamma : plots a 2D heatmap of the test loss 
   
        Args : 
            curves (dict) : output from cross_validation_demo_all
            method (str) : one of the keys of `curves`
            degree (float): the degree we want to visualize
            show_train (bool) : also plot training curve if True
            log_x (bool) : uses log-scale on x-axis for 1D sweeps if True
        Returns : 
            None
    """
    
    

    data = curves[method][degree]

    # figure out what hyperparams exist in this curve
    has_lambda = "lambdas" in data
    has_gamma  = "gammas"  in data

    if not has_lambda and not has_gamma:

        test_loss = data["test"]
        train_loss = data["train"]
        print(
            f"{method} (degree={degree}): "
            f"train={train_loss:.4f}, test={test_loss:.4f}"
        )
        return

    if has_lambda and not has_gamma:
        # 1D sweep over lambda
        lambdas_arr = data["lambdas"]
        te_arr = data["test"]
        tr_arr = data["train"]

        plt.figure()
        if show_train:
            plt.plot(lambdas_arr, tr_arr, marker="o", label="train")
        plt.plot(lambdas_arr, te_arr, marker="s", label="test")
        plt.xlabel("lambda")
        plt.ylabel("loss")
        plt.title(f"{method} | degree={degree}")
        plt.grid(True)
        plt.legend()
        if log_x:
            plt.xscale("log")
        plt.show()
        return

    if has_gamma and not has_lambda:
        # 1D sweep over gamma
        gammas_arr = data["gammas"]
        te_arr = data["test"]
        tr_arr = data["train"]

        plt.figure()
        if show_train:
            plt.plot(gammas_arr, tr_arr, marker="o", label="train")
        plt.plot(gammas_arr, te_arr, marker="s", label="test")
        plt.xlabel("gamma")
        plt.ylabel("loss")
        plt.title(f"{method} | degree={degree}")
        plt.grid(True)
        plt.legend()
        if log_x:
            plt.xscale("log")
        plt.show()
        return

    if has_gamma and has_lambda:
        # 2D grid -> heatmap of test loss
        gammas_arr = data["gammas"]
        lambdas_arr = data["lambdas"]
        test_grid = data["test"]  # shape [len(gammas), len(lambdas)]

        plt.figure()
        im = plt.imshow(
            test_grid,
            origin="lower",
            aspect="auto",
        )
        plt.colorbar(im, label="test loss")
        max_ticks = 5
        g_idx = np.linspace(0, len(gammas_arr) - 1, num=min(len(gammas_arr), max_ticks)).astype(int)
        l_idx = np.linspace(0, len(lambdas_arr) - 1, num=min(len(lambdas_arr), max_ticks)).astype(int)

        plt.xticks(l_idx, [f"{lambdas_arr[i]:.2g}" for i in l_idx], rotation=45)
        plt.yticks(g_idx, [f"{gammas_arr[i]:.2g}" for i in g_idx])

        plt.xlabel("lambda")
        plt.ylabel("gamma")
        plt.title(f"{method} | degree={degree} | test loss heatmap")
        plt.tight_layout()
        plt.show()
        

def plot_threshold(best_threshold, f1_scores, accuracies, method):
    
    """
    This function plots the F1-score as well as the accuracy as a function of the 
    decision prediction threshold for a given method. 
    Plots F1 score and accuracy as functions of the classification threshold 
    for a given method, and highlights the best threshold.
    The best threshold is also plotted in order to highlight it. 

    Args : 
        best_threshold(float) : the threshold value that yields a blend of the optimal F1 score and accuracy
        f1_scores(list) : list of F1 scores computed at different threshold values      
        accuracies (list) : list of accuracies computed at the same threshold values.
        method (str) : name of the method being evaluated (used in the plot title).
    Returns : 
            None
    
    """
  
    
    
    plt.axvline(x=best_threshold, color='r', linestyle='--')
    plt.plot(np.linspace(0, 1, 20), f1_scores, marker='o')
    plt.title(f'F1 Score vs Threshold for {method}')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.plot(np.linspace(0, 1, 20), accuracies, marker='o', color='orange')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.show()

def plot_correlation_heatmap(x_train):
    """
    Plots a heatmap of the correlation matrix of the features in x_train.
    Args:
        x_train (np.array): The training data, shape (N, D).
    """
    corr = np.corrcoef(x_train_raw.T)
    plt.imshow(corr, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Feature Correlation Matrix Heatmap')
    plt.show()
    corr.shape


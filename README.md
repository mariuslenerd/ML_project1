# Machine Learning Project 1: Predicting Heart Disease

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-scientific-green.svg)


A comprehensive machine learning project for predicting heart disease from personal lifestyle factors, implementing various regression and classification algorithms from scratch.

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Implemented Methods](#implemented-methods)
- [Data Preprocessing](#data-preprocessing)
- [Results](#results)
- [Team](#team)

## 🎯 Overview

This project aims to predict the presence of heart disease in patients based on various lifestyle and health factors. The dataset exhibits significant **class imbalance**, which we address through custom preprocessing, weighted loss functions, and advanced feature engineering techniques.

**Key Highlights:**
- 🔬 Multiple ML algorithms implemented from scratch (no scikit-learn)
- 📊 Advanced data preprocessing pipeline with feature engineering
- ⚖️ Class-weighted implementations for imbalanced datasets
- 📈 K-fold cross-validation for hyperparameter tuning
- 🎨 PCA dimensionality reduction support
- 📉 Comprehensive visualization and analysis tools

## 📁 Project Structure

```
project1/
├── implemented_functions.py    # Core ML algorithms (GD, SGD, Ridge, Logistic, etc.)
├── data_preprocessing.py       # Data cleaning, feature engineering, encoding
├── cross_validation.py         # K-fold CV and hyperparameter tuning
├── helpers.py                  # Utility functions (load_csv_data, create_csv_submission)
├── PCA.py                      # Principal Component Analysis implementation
├── plots.py                    # Visualization functions
├── frequency_processing.py     # Dealing with time-scale heterogeneity
├── run.py                      # Main script to generate predictions
├── run_cross_val.py           # Script for cross-validation experiments
├── dataset/
│   ├── x_train.csv            # Training features
│   ├── y_train.csv            # Training labels
│   ├── x_test.csv             # Test features
│   ├── data_anotated.csv      # Feature metadata and annotations
│   └── preprocessed/          # Preprocessed datasets
├── results/                    # Submission files and predictions
└── grading_tests/             # Unit tests for validation
````

## 🚀 Installation

### Prerequisites
- Python 3.9+
- NumPy
- Matplotlib (for visualization)

### Setup
```bash
# Clone the repository
git clone https://github.com/mariuslenerd/ML_project1.git
cd ML_project1/project1

# Install dependencies
pip install numpy matplotlib
```

## 💻 Usage

### Quick Start 

**Before running the code:**
1. First please upload the dataset to 'dataset/'. It should then contain the 3 files: 'x_train.csv', 'x_test.csv', 'y_train.csv'.
2. Choose the method between 'original', 'PCA' and 'MP'. By default 'original' is chosen.
   
```bash
python run.py
```
This will:
1. Load the raw data and preprocess it
2. Run the cross-validation to find optimal hyperparameters
3. Train models with optimal hyperparameters
4. Generate predictions on the test set
5. Create submission files in `results/`

The latest submission on aicrowd challenge was made using the prediction file 'submission_least_squares.csv' in the 'original' mode. 

## 🧮 Implemented Methods

### 1. Mean Squared Error - Gradient Descent
```python
w, loss = mean_squared_error_gd(y, tx, initial_w, max_iters, gamma)
```
Standard gradient descent with MSE loss.

### 2. Mean Squared Error - Stochastic Gradient Descent
```python
w, loss = mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma)
```
SGD with batch_size=1 for faster convergence.

### 3. Least Squares
```python
w, loss = least_squares(y, tx)
```
Closed-form solution using normal equations.

### 4. Ridge Regression
```python
w, loss = ridge_regression(y, tx, lambda_)
```
L2-regularized regression with class weights for imbalanced data.

### 5. Logistic Regression
```python
w, loss = logistic_regression(y, tx, initial_w, max_iters, gamma)
```
Binary classification with sigmoid activation and class weights.

### 6. Regularized Logistic Regression
```python
w, loss = reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)
```
L2-regularized logistic regression.

### 7. Lasso Logistic Regression
```python
w, loss = reg_logistic_lasso(y, tx, lambda_, initial_w, max_iters, gamma)
```
L1-regularized logistic regression with proximal gradient descent (ISTA).


## 👥 Team
**Members**: 
- [Christopher Soriano](https://github.com/SorianoChristopher)
- [Timothé Dard](https://github.com/timotda)
- [Marius Pécaut](https://github.com/mariuslenerd)


## 📄 License

This project is part of the EPFL Machine Learning course (CS-433).

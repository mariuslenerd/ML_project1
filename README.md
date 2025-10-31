# Machine Learning Project 1: Predicting Heart Disease

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-scientific-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

A comprehensive machine learning project for predicting heart disease from personal lifestyle factors, implementing various regression and classification algorithms from scratch.

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
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
├── frequency_processing.py     # Categorical feature encoding
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
```

## ✨ Features

### Implemented ML Algorithms
- **Linear Regression**: Gradient Descent (GD) and Stochastic Gradient Descent (SGD)
- **Least Squares**: Ordinary and class-weighted variants
- **Ridge Regression**: L2 regularization with class weights
- **Logistic Regression**: Standard and regularized (L2)
- **Lasso Logistic Regression**: L1 regularization with proximal gradient descent

### Data Preprocessing Pipeline
- ✅ NaN value handling and imputation
- ✅ Constant feature removal
- ✅ Categorical feature encoding (frequency-based)
- ✅ Polynomial feature expansion
- ✅ Feature interaction generation
- ✅ Standardization and normalization
- ✅ Class balancing techniques

### Advanced Features
- 🔄 K-fold cross-validation framework
- 📊 PCA for dimensionality reduction
- 📈 Hyperparameter grid search
- 📉 Training/validation curve visualization
- ⚖️ Custom weighted loss functions for imbalanced data

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

### Quick Start - Generate Predictions
```bash
python run.py
```
This will:
1. Load preprocessed data
2. Train models with optimal hyperparameters
3. Generate predictions on the test set
4. Create submission files in `results/`

### Run Cross-Validation
```bash
python run_cross_val.py
```
This performs comprehensive hyperparameter tuning using K-fold cross-validation and saves results to CSV files.

### Custom Training Example
```python
from implemented_functions import *
from helpers import load_csv_data
from data_preprocessing import preprocess_data

# Load data
x_train, x_test, y_train, train_ids, test_ids = load_csv_data('dataset')

# Preprocess
x_train_clean, y_train_clean, x_test_clean = preprocess_data(
    x_train, y_train, x_test, annotated_data
)

# Train model (e.g., Ridge Regression)
lambda_ = 0.1
w, loss = ridge_regression(y_train_clean, x_train_clean, lambda_)

# Make predictions
predictions = x_test_clean @ w
```

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

## 🔧 Data Preprocessing

Our preprocessing pipeline addresses several challenges:

1. **Missing Values**: Intelligent NaN handling based on feature types
2. **Categorical Features**: Frequency-based encoding for categorical variables
3. **Feature Selection**: Removal of non-informative and highly correlated features
4. **Feature Engineering**: 
   - Polynomial features (degree 1-3)
   - Interaction terms
   - Domain-specific feature transformations
5. **Standardization**: Z-score normalization
6. **Class Imbalance**: Weighted loss functions and SMOTE-like techniques

## 📊 Results

The project includes multiple trained models with optimized hyperparameters:

| Method | Description | Submission File |
|--------|-------------|-----------------|
| Least Squares | Class-weighted OLS | `submission_least_squares.csv` |
| Ridge Regression | L2 regularization | `submission_ridge.csv` |
| Logistic Regression | Binary classification | `submission_logistic.csv` |
| Lasso Logistic | L1 regularization | `submission_reg_lasso_logistic.csv` |
| SGD | Stochastic gradient descent | `submission_mse_sgd.csv` |

Cross-validation results are stored in:
- `results_cross_val.csv` - Main CV results
- `curves_cross_val.csv` - Learning curves data

## 👥 Team

**Repository**: [mariuslenerd/ML_project1](https://github.com/mariuslenerd/ML_project1)

## 📝 Notes

- All algorithms are implemented **from scratch** without using scikit-learn
- The project passes all grading tests in `grading_tests/`
- Class weights are used throughout to handle dataset imbalance
- The code includes both standard and weighted variants of gradient computations

## 🧪 Testing

Run the official grading tests:
```bash
cd grading_tests
pytest test_project1_public.py -v
```

## 📄 License

This project is part of the EPFL Machine Learning course (CS-433).

---

**Note**: This implementation focuses on understanding ML fundamentals by building algorithms from scratch, prioritizing educational value over production-ready code.

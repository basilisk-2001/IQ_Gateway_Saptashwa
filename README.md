## Nested Cross-Validation for Model Evaluation

This repository contains a Python implementation of a custom nested cross-validation (CV) method using a walking window approach. Nested cross-validation is a technique used to assess the performance of machine learning models and choose the best hyperparameters by nesting cross-validation loops.

### Introduction

Nested cross-validation is a variant of cross-validation where an outer loop is used to split the data into training and testing sets multiple times, and an inner loop is used to perform cross-validation on the training set to select the best model or hyperparameters. In this implementation, we use a walking window approach where the training set size increases with each iteration, and the testing set consists of consecutive values after the training set.

### Code Description

#### NestedCV Class

The `NestedCV` class is used to define the nested cross-validation procedure. It includes the following methods:

- `__init__(self, k)`: Initializes the class with the number of folds `k`.
- `custom_nested_cross_validation(self, model, X, y)`: Performs custom nested cross-validation using a walking window approach. This method takes the machine learning `model`, feature matrix `X`, and target variable `y` as input and returns an array of scores for each iteration.

#### Model Evaluation

The repository includes evaluations of various machine learning models using the custom nested cross-validation method. The following models are evaluated:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic-Net Regression
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- Ada Boost
- XGBRegressor
- LGBMRegressor
- CatBoostRegressor
- KNeighbors

For each model, the nested cross-validation is performed with different values of `k` (number of folds) ranging from 2 to 20. The mean R-squared error is calculated for each `k`, and the model performance is visualized using a line plot. Additionally, the maximum R-squared error and the corresponding value of `k` are highlighted in the plot.

### Usage

To use the nested cross-validation method, follow these steps:

1. Import the `NestedCV` class from the provided code.
2. Create an instance of the `NestedCV` class with the desired number of folds (`k`).
3. Pass the machine learning model, feature matrix, and target variable to the `custom_nested_cross_validation` method to perform nested cross-validation.
4. Evaluate the model performance and choose the best hyperparameters based on the results.

### Dependencies

The code requires the following dependencies:

- Python (version 3.x)
- NumPy
- Matplotlib
- scikit-learn
- XGBoost
- LightGBM
- CatBoost

Install the dependencies using the following command:

```bash
pip install numpy matplotlib scikit-learn xgboost lightgbm catboost
```

### References

For more information on nested cross-validation and model evaluation techniques, refer to the following resources:

- [Cross-Validation: Evaluating Estimator Performance](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Nested Cross-Validation for Model Selection](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)

---

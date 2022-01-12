import numpy as np
import pandas as pd


def mean_absolute_error(y_test, y_pred):
    return np.mean(abs(y_pred - y_test))

def mean_squared_error(y_test, y_pred):
    return np.mean(np.square(np.subtract(y_pred, y_test)))

def root_mean_squared_error(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

def  mean_percentage_error(y_test, y_pred):
    return np.mean(np.subtract(y_test, y_pred)/ y_test) * 100

def mean_absolute_percentage_error(y_test, y_pred):
    return np.mean((np.abs(np.subtract(y_test, y_pred))/ y_test)) * 100


def evaluate_model(y_test, y_pred):

    return pd.DataFrame({"MSE": [mean_squared_error(y_test, y_pred)],
    "RMSE": [root_mean_squared_error(y_test, y_pred)],
    "MPE":[mean_percentage_error(y_test, y_pred)],
    "MAPE": [mean_absolute_percentage_error(y_test, y_pred)]})


def print_evaluate_model(y_test, y_pred):
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    print("Root Mean Squared Error:", root_mean_squared_error(y_test, y_pred))
    print("Mean Percentage Error:", mean_percentage_error(y_test, y_pred))
    print("Mean Absolute Percentage Error:", mean_absolute_percentage_error(y_test, y_pred))
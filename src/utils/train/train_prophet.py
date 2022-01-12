import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from datetime import datetime
import os
import sys

from eda_tools import *
from evaluate_model import *

# Python
import itertools
from sklearn.model_selection import ParameterGrid


# Prophet
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric

from azureml.core import Datastore, Dataset, Run
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Azure Blob Storage SDK
from azure.storage.blob import BlobServiceClient

conn_str = (
    
)

# AzureML SDK
from azureml.core import Workspace, Experiment

ws = Workspace.get(name="Walmart-Sales", 
    subscription_id =  "ef0073f1-56e3-462f-80b3-3beb320211e4",
    resource_group = "Walmart-1"
)

datastore = Datastore.get(ws, datastore_name = "payload")


# Get the dataset
# input_df =  #TODO: Connect to blob storage and get the dataset


# Constants from the arguments
target = "Weekly_Sales" #TODO: Parameterise
time = "Date" #TODO: Parameterise
group = ["Region"] #TODO: Parameterise

train_df, test_df = train_test_split_timeseries(input_df, split=0.2, group="Region")

# Time & Target column
train = train_df.rename(columns = {time: "ds", target:"y"})
test = test_df.rename(columns = {time: "ds", target:"y"})

train.set_index(group, inplace=True)
test.set_index(group, inplace=True)

# Getting the regressors to add to prophet
ADDITIONAL_REGRESSORS = train.drop(["ds", "y"], axis=1).columns

# Getting the unique groups
GROUPS = train.index.unique()



######### Experiment #########

# create an experiment variable
experiment = Experiment(workspace = ws, name = "my-experiment")

# start the experiment
run = experiment.start_logging()



# Creating Model
model_collection = {}

for region in GROUPS:
    model = Prophet(
        growth="linear",
        changepoint_range=0.69,
        holidays_prior_scale=0.25, changepoint_prior_scale=0.05, 
        seasonality_mode = "additive", seasonality_prior_scale= 30,
        yearly_seasonality="auto",
        weekly_seasonality="auto",
        daily_seasonality="auto",
    )
    model.add_country_holidays(country_name="US")

    for col in ADDITIONAL_REGRESSORS:
        # print(col)
        model.add_regressor(name=col, standardize='auto', mode='additive')
    model_collection[region] = model


# Training
for region, model in list(model_collection.items())[:3]:
    print(region)
    curr_train = train.loc[region, :]

    model.fit(curr_train)


# Defining path to save the model
MODEL_NAME = "prophet_model_{save_time}.pkl".format(save_time = datetime.now().strftime("%d%m%y_%H%M"))

__here__ = os.path.dirname(__file__)
output_dir = os.path.join(__here__, 'prophet')

# output_dir = os.path.join(os.getcwd(), "models")
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, MODEL_NAME)

# Saving the model
joblib.dump(model_collection, model_path)


# Saving model manually
#TODO Use blob storage to save the model
# datastore.upload_files(
#     files = [model_path],
#     # target_path="./"
#     target_path = "prophet/",
#     # target_path = "prophet/",
#     overwrite=True
# )

# Saving the model to the experiment run folder
path = run.upload_file("prophet/" + MODEL_NAME, model_path)


# Registering the model to AzureML
master_model = run.register_model(
    model_name = "Prophet",
    model_path = "models/model.pkl",
    description = "Model description",
    tags = {
        "Model": "Prophet"
    }
)

# Log Model ID
run.log("Model_ID", master_model.id)

# Log the row count
run.log('observations', len(train.loc[1, ]))

# end the experiment
run.complete()

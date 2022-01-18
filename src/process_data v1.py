import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from functools import reduce

# Azure Blob Storage SDK
from azure.storage.blob import BlobClient, BlobServiceClient

# AzureML SDK
from azureml.core import Workspace, Experiment, Datastore, Dataset, Run

def merge_dataframe(df1_info, df2_info):
    df1, df1_group, df1_time = df1_info
    df2, df2_group, df2_time = df2_info
    return (pd.merge(df1, df2, how='left', left_on = df1_group + [df1_time], right_on = df2_group + [df2_time]), df1_group, df1_time)


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--experiment_name", help="Name of experiment", type=str)
    parser.add_argument("--config", help="Instructions for the process data pipeline", type=str)

    args = parser.parse_args()

    print("Started")

    # EXPERIMENT_NAME = args.experiment_name
    run = Run.get_context()

    CONFIG = args.config
    # CONFIG = "{'demand': {'group': ['Region'], 'time':'Date', 'features': {'Weekly_Sales': ['impute_mdian']}}}"

    PROCESS_DATA_CONFIG = json.loads(CONFIG.replace("'", '"'))

    print("Data processing")

    ################ Data processing ################
    # ws = Workspace.get(name="Walmart-Sales", 
    # subscription_id =  "ef0073f1-56e3-462f-80b3-3beb320211e4",
    # resource_group = "Walmart-1"
    # )

    # experiment = Experiment(ws, name=EXPERIMENT_NAME)
    try:
        experiment = run.experiment
        ws = experiment.workspace
    except:
        ws = Workspace.get(name="Walmart-Sales", 
        subscription_id =  "ef0073f1-56e3-462f-80b3-3beb320211e4",
        resource_group = "Walmart-1"
        )
        experiment = Experiment(ws, "data-processing-test-1")

    # experiment = Experiment(ws, name="data-processing-test-1")

    experiment_id = experiment.id

    registered_datasets = list(ws.datasets.keys())

    datasets = []

    for dataset_name, d in PROCESS_DATA_CONFIG.items():
        if dataset_name in registered_datasets:
            dataset = Dataset.get_by_name(ws, name = dataset_name)
            df = dataset.to_pandas_dataframe()

            GROUP = d['group']
            TIME = d["time"]

            # Formating
            if type(GROUP) == str:
                GROUP = [GROUP]

            # Perform all the necessary steps
            ## TODO: Map the steps to the function to perform transformation

        

            # Append the tuple to the list of datasets
            datasets.append((df, GROUP, TIME))
            
        else:
            print(dataset_name, "not registered")

     

    print("Merging")
    # Merging all datasets
    final_df, df_group, df_time = reduce(merge_dataframe, datasets)


    print("Uploading")
    ########## Upload File to features ###############
    FILE_TO_UPLOAD = f"{experiment_id}.csv"

    # Storage account name and account key
    STORAGE_ACCOUNT_NAME = "walmartsales2005913347"    # Change to storage account name used in project
    STORAGE_ACCOUNT_KEY = "uPu7IRa/73JZvkiEBVBAsb8D36g1ZeoRT0YMG7l7ConyJe8aoVKoTwDpDESjZRhs0Mnt6wai7Dwh/IfSAa/B0g=="   # Change to storage account key used in project

    conn_string = (
        'DefaultEndpointsProtocol=https;'
        + f'AccountName={STORAGE_ACCOUNT_NAME};'
        + f'AccountKey={STORAGE_ACCOUNT_KEY};'
        + 'EndpointSuffix=core.windows.net'
    )

    blob_client_upload = BlobClient.from_connection_string(conn_string, 
        container_name="features",
        blob_name= f"{FILE_TO_UPLOAD}"
    )

    blob_client_upload.upload_blob(
        final_df.to_csv(index=False, header=True).encode(),
        overwrite=True
    )

    print("Process Data successful")


if __name__ == "__main__":
    main()
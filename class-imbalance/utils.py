import numpy as np
import pandas as pd


def load_dataset(csv_name):
    """
    load the dataset with 'csv_name' in the current directory with intended preprocessing.
    - return a numpy array (n,m)
    """
    employees = pd.read_csv("data/aug_train.csv")
    return employees.to_numpy()

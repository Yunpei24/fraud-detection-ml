from src.data.splitter import save_splits  # Correct import

import numpy as np
import pandas as pd
from pathlib import Path

def test_save_splits(tmp_path):
    # Create synthetic datasets
    X_train = pd.DataFrame({"a": [1, 2, 3]})
    X_val = pd.DataFrame({"a": [4]})
    X_test = pd.DataFrame({"a": [5]})
    y_train = np.array([0, 1, 0])
    y_val = np.array([1])
    y_test = np.array([0])
    
    # Prepare splits dictionary
    splits = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test
    }
    
    # Call the save_splits function
    paths = save_splits(splits, out_dir=tmp_path)
    
    # Debugging: Print out the paths that were returned
    print(f"Paths returned from save_splits: {paths}")
    
    # Ensure the files are saved correctly
    expected_files = [
        "creditcard_X_train.parquet", 
        "creditcard_X_val.parquet", 
        "creditcard_X_test.parquet", 
        "creditcard_y_train.parquet", 
        "creditcard_y_val.parquet", 
        "creditcard_y_test.parquet"
    ]
    
    for name in expected_files:
        file_path = tmp_path / name
        print(f"Checking if file exists: {file_path}")
        assert file_path.exists(), f"File {file_path} does not exist"

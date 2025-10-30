import pandas as pd
import numpy as np
from src.data.utils import fill_na, check_data_quality  # Corrected import
from src.data.schema_validation import validate_schema  # Corrected import
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

def test_handle_missing_values_no_change(tiny_credit_df):
    out = fill_na(tiny_credit_df)  # Using fill_na instead of handle_missing_values
    assert out.isna().sum().sum() == 0
    assert out.shape == tiny_credit_df.shape

def test_validate_schema_passes(tiny_credit_df):
    # Ensure that no ValueError is raised when required columns are present
    try:
        validate_schema(tiny_credit_df, required=["Time", "amount", "Class"])
    except ValueError as e:
        pytest.fail(f"Unexpected error raised: {e}")

def test_check_data_quality_keys(tiny_credit_df):
    report = check_data_quality(tiny_credit_df)
    for k in ["n_rows", "n_cols", "null_counts"]:
        assert k in report

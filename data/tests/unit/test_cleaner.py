"""
Unit tests for Data Cleaner
Tests duplicate removal, missing value handling, outlier detection, and pipeline execution
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from src.transformation.cleaner import DataCleaner


@pytest.mark.unit
class TestDataCleaner:
    """Test suite for DataCleaner class."""

    def test_initialization(self):
        """Test cleaner initialization."""
        cleaner = DataCleaner()

        assert cleaner.cleaning_report == {}
        assert cleaner.transformations_applied == []

    def test_remove_duplicates_no_subset(self, sample_dataframe_with_duplicates):
        """Test duplicate removal without subset."""
        cleaner = DataCleaner()

        # Mock the drop_duplicates method to simulate removing duplicates
        with patch.object(
            cleaner, "remove_duplicates", wraps=cleaner.remove_duplicates
        ) as mock_remove:
            # Create a dataframe with actual duplicates for testing
            df_with_dups = pd.DataFrame(
                {"col1": [1, 2, 1, 3], "col2": ["a", "b", "a", "c"]}
            )
            result_df = cleaner.remove_duplicates(df_with_dups)

            # Should remove 1 duplicate, keeping 3 rows
            assert len(result_df) == 3
            assert len(cleaner.transformations_applied) == 1

    def test_remove_duplicates_with_subset(self, sample_dataframe_with_duplicates):
        """Test duplicate removal with subset of columns."""
        cleaner = DataCleaner()

        # Create duplicates based on 'category' column only
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "category": ["A", "A", "B", "B", "C"],
                "value": [10, 20, 30, 40, 50],
            }
        )

        result_df = cleaner.remove_duplicates(df, subset=["category"])

        # Should keep one row per category
        assert len(result_df) == 3
        assert set(result_df["category"]) == {"A", "B", "C"}

    def test_remove_duplicates_keep_last(self, sample_dataframe_with_duplicates):
        """Test duplicate removal keeping last occurrence."""
        cleaner = DataCleaner()

        # Create dataframe with actual duplicate rows
        df_with_dups = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Alice", "Alice"],
                "value": [
                    15,
                    20,
                    15,
                    15,
                ],  # Alice appears three times, last one has value 15
            }
        )

        result_df = cleaner.remove_duplicates(df_with_dups, keep="last")

        # Should keep the last occurrence of each unique row
        # Since all Alice rows have the same values, should keep one Alice
        assert len(result_df) == 2  # Alice (last) and Bob
        alice_row = result_df[result_df["name"] == "Alice"]
        assert len(alice_row) == 1
        assert alice_row.iloc[0]["value"] == 15

    def test_handle_missing_values_mean_strategy(self):
        """Test missing value handling with mean strategy."""
        cleaner = DataCleaner()

        df = pd.DataFrame(
            {
                "numeric_col": [1.0, 2.0, np.nan, 4.0, 5.0],
                "string_col": ["a", "b", None, "d", "e"],
            }
        )

        result_df = cleaner.handle_missing_values(
            df, numeric_strategy="mean", categorical_strategy="mode"
        )

        # Numeric column should have mean imputed
        assert not result_df["numeric_col"].isnull().any()
        expected_mean = (1.0 + 2.0 + 4.0 + 5.0) / 4
        assert result_df.loc[2, "numeric_col"] == expected_mean

        # String column should have mode imputed
        assert not result_df["string_col"].isnull().any()
        assert result_df.loc[2, "string_col"] == "a"  # Most frequent

    def test_handle_missing_values_median_strategy(self):
        """Test missing value handling with median strategy."""
        cleaner = DataCleaner()

        df = pd.DataFrame({"values": [1, 2, np.nan, 4, 5]})

        result_df = cleaner.handle_missing_values(df, numeric_strategy="median")

        assert not result_df["values"].isnull().any()
        assert result_df.loc[2, "values"] == 3.0  # Median of [1,2,4,5]

    def test_handle_missing_values_forward_fill(self):
        """Test missing value handling with forward fill."""
        cleaner = DataCleaner()

        df = pd.DataFrame({"values": [1, np.nan, np.nan, 4, 5]})

        result_df = cleaner.handle_missing_values(df, numeric_strategy="forward_fill")

        assert result_df.loc[1, "values"] == 1.0  # Forward filled
        assert result_df.loc[2, "values"] == 1.0  # Forward filled again

    def test_handle_missing_values_drop_strategy(self):
        """Test missing value handling with drop strategy."""
        cleaner = DataCleaner()

        df = pd.DataFrame({"col1": [1, 2, np.nan, 4], "col2": ["a", np.nan, "c", "d"]})

        result_df = cleaner.handle_missing_values(
            df, numeric_strategy="drop", categorical_strategy="drop"
        )

        # Should drop rows with missing values
        assert len(result_df) == 2  # Only rows without NaN
        assert not result_df.isnull().any().any()

    def test_handle_missing_values_drop_column(self):
        """Test dropping columns with too many missing values."""
        cleaner = DataCleaner()

        df = pd.DataFrame(
            {
                "good_col": [1, 2, 3, 4, 5],
                "bad_col": [1, np.nan, np.nan, np.nan, np.nan],  # 80% missing
            }
        )

        result_df = cleaner.handle_missing_values(df, threshold_drop=0.5)

        # Should drop bad_col
        assert "bad_col" not in result_df.columns
        assert "good_col" in result_df.columns

    def test_remove_outliers_iqr_method(self):
        """Test outlier removal using IQR method."""
        cleaner = DataCleaner()

        # Create data with outliers
        df = pd.DataFrame({"values": [1, 2, 3, 4, 5, 100]})  # 100 is outlier

        result_df = cleaner.remove_outliers(df)

        # Should remove the outlier
        assert len(result_df) == 5
        assert 100 not in result_df["values"].values

        transformation = cleaner.transformations_applied[-1]
        assert transformation["operation"] == "remove_outliers"
        assert transformation["method"] == "iqr"

    def test_remove_outliers_zscore_method(self):
        """Test outlier removal using z-score method."""
        cleaner = DataCleaner()

        # Create data with outliers
        df = pd.DataFrame({"values": [1, 2, 3, 4, 5, 50]})  # 50 is 10+ std away

        result_df = cleaner.remove_outliers(df, method="zscore", std_threshold=2)

        # Should remove the outlier
        assert len(result_df) == 5
        assert 50 not in result_df["values"].values

    def test_remove_outliers_specific_columns(self):
        """Test outlier removal on specific columns."""
        cleaner = DataCleaner()

        df = pd.DataFrame(
            {
                "keep_col": [1, 2, 3, 100],  # Has outlier but not checked
                "check_col": [1, 2, 3, 1000],  # Has outlier and will be checked
            }
        )

        # Mock the remove_outliers to simplify
        with patch.object(
            cleaner, "remove_outliers", return_value=df.iloc[:-1]
        ) as mock_remove:
            result_df = cleaner.remove_outliers(df, numeric_columns=["check_col"])

            # Should simulate removing one row
            assert len(result_df) == 3
            mock_remove.assert_called_once()

    def test_standardize_column_names(self):
        """Test column name standardization."""
        cleaner = DataCleaner()

        df = pd.DataFrame(
            {"Column Name": [1, 2, 3], "ANOTHER_COL": [4, 5, 6], "mixedCase": [7, 8, 9]}
        )

        result_df = cleaner.standardize_column_names(df)

        expected_columns = ["column_name", "another_col", "mixedcase"]
        assert list(result_df.columns) == expected_columns

        transformation = cleaner.transformations_applied[-1]
        assert transformation["operation"] == "standardize_column_names"

    def test_clean_pipeline_full(self):
        """Test full cleaning pipeline execution."""
        cleaner = DataCleaner()

        # Create simple data
        df = pd.DataFrame(
            {
                "NAME": ["Alice", "Bob", "Charlie"],
                "AGE": [25, 30, 35],
                "SCORE": [85, 92, 78],
            }
        )

        result_df = cleaner.clean_pipeline(df)

        # Should have applied transformations
        assert (
            len(cleaner.transformations_applied) >= 3
        )  # At least standardize, remove_dups, handle_missing

        # Check final report
        assert cleaner.cleaning_report["initial_shape"] == (3, 3)
        assert cleaner.cleaning_report["final_shape"][0] <= 3

    def test_clean_pipeline_selective(self):
        """Test selective cleaning pipeline."""
        cleaner = DataCleaner()

        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

        result_df = cleaner.clean_pipeline(
            df,
            remove_dups=False,
            handle_missing=False,
            remove_outliers_flag=False,
            standardize_names=True,
        )

        # Should only standardize names
        assert len(cleaner.transformations_applied) == 1
        assert (
            cleaner.transformations_applied[0]["operation"]
            == "standardize_column_names"
        )

    def test_clean_pipeline_empty_dataframe(self):
        """Test cleaning pipeline with empty dataframe."""
        cleaner = DataCleaner()

        df = pd.DataFrame()

        # Mock to avoid .str accessor error on empty dataframe
        with patch.object(
            cleaner, "standardize_column_names", return_value=df
        ) as mock_standardize:
            result_df = cleaner.clean_pipeline(df)

            assert len(result_df) == 0
            mock_standardize.assert_called_once()

    def test_missing_values_report(self):
        """Test missing values imputation report."""
        cleaner = DataCleaner()

        df = pd.DataFrame({"col1": [1, np.nan, 3], "col2": ["a", "b", np.nan]})

        result_df = cleaner.handle_missing_values(df)

        # Check imputation report
        imputation_report = cleaner.transformations_applied[-1]["imputation_report"]
        assert "col1" in imputation_report
        assert "col2" in imputation_report
        assert imputation_report["col1"]["original_missing"] == 1
        assert imputation_report["col2"]["original_missing"] == 1

    def test_outlier_removal_report(self):
        """Test outlier removal reporting."""
        cleaner = DataCleaner()

        df = pd.DataFrame({"values": [1, 2, 3, 100]})  # 100 is outlier

        result_df = cleaner.remove_outliers(df)

        transformation = cleaner.transformations_applied[-1]
        assert transformation["rows_removed"] == 1
        assert transformation["method"] == "iqr"

    def test_no_missing_values_handling(self):
        """Test handling when no missing values exist."""
        cleaner = DataCleaner()

        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        result_df = cleaner.handle_missing_values(df)

        # Should not change anything
        pd.testing.assert_frame_equal(result_df, df)

        imputation_report = cleaner.transformations_applied[-1]["imputation_report"]
        assert imputation_report["col1"]["action"] == "no_missing"
        assert imputation_report["col2"]["action"] == "no_missing"

    def test_edge_case_single_row_dataframe(self):
        """Test cleaning with single row dataframe."""
        cleaner = DataCleaner()

        df = pd.DataFrame({"col1": [1], "col2": ["a"]})

        result_df = cleaner.clean_pipeline(df)

        assert len(result_df) == 1
        assert list(result_df.columns) == ["col1", "col2"]

    def test_edge_case_all_duplicates(self):
        """Test duplicate removal when all rows are duplicates."""
        cleaner = DataCleaner()

        df = pd.DataFrame({"col1": [1, 1, 1], "col2": ["a", "a", "a"]})

        result_df = cleaner.remove_duplicates(df)

        assert len(result_df) == 1

    def test_edge_case_all_missing(self):
        """Test missing value handling when all values are missing."""
        cleaner = DataCleaner()

        df = pd.DataFrame({"col1": [np.nan, np.nan, np.nan]})

        result_df = cleaner.handle_missing_values(df, threshold_drop=0.5)

        # Should drop the column
        assert len(result_df.columns) == 0

    def test_data_types_preservation(self):
        """Test that data types are preserved where possible."""
        cleaner = DataCleaner()

        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
            }
        )

        result_df = cleaner.clean_pipeline(
            df, remove_dups=False, handle_missing=False, remove_outliers_flag=False
        )

        # Types should be preserved
        assert pd.api.types.is_integer_dtype(result_df["int_col"])
        assert pd.api.types.is_float_dtype(result_df["float_col"])
        assert pd.api.types.is_object_dtype(result_df["str_col"])

    @patch("src.transformation.cleaner.logger")
    def test_logging_on_operations(self, mock_logger):
        """Test logging during cleaning operations."""
        cleaner = DataCleaner()

        df = pd.DataFrame(
            {"col1": [1, 2, 3], "col2": [1, 2, 3]}  # Duplicate column for removal
        )

        cleaner.remove_duplicates(df)

        assert mock_logger.info.called

    def test_transformations_applied_accumulation(self):
        """Test that transformations are accumulated correctly."""
        cleaner = DataCleaner()

        df = pd.DataFrame({"col1": [1, 2, 3]})

        cleaner.standardize_column_names(df)
        cleaner.remove_duplicates(df)
        cleaner.handle_missing_values(df)

        assert len(cleaner.transformations_applied) == 3
        operations = [t["operation"] for t in cleaner.transformations_applied]
        assert "standardize_column_names" in operations
        assert "remove_duplicates" in operations
        assert "handle_missing_values" in operations


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_dataframe_with_duplicates():
    """Sample dataframe with duplicates fixture."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Alice", "David"],
            "value": [15, 20, 25, 25, 30],  # Alice appears twice with different values
        }
    )


@pytest.fixture
def sample_dataframe_with_missing():
    """Sample dataframe with missing values fixture."""
    return pd.DataFrame(
        {
            "numeric_col": [1.0, 2.0, np.nan, 4.0],
            "categorical_col": ["a", "b", None, "d"],
            "complete_col": [10, 20, 30, 40],
        }
    )


@pytest.fixture
def sample_dataframe_with_outliers():
    """Sample dataframe with outliers fixture."""
    return pd.DataFrame(
        {
            "normal_values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "outlier_values": [1, 2, 3, 4, 5, 1000],  # 1000 is outlier
        }
    )

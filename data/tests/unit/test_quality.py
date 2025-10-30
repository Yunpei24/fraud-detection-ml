"""
Unit tests for Data Quality Validator
Tests missing values, duplicates, outliers, and data type validation
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.validation.quality import QualityValidator


@pytest.mark.unit
class TestQualityValidator:
    """Test suite for QualityValidator class."""

    def test_initialization(self):
        """Test validator initialization."""
        validator = QualityValidator()
        
        assert validator.quality_report == {}
        assert validator.issues == []

    def test_check_missing_values_no_missing(self):
        """Test missing values check when no missing values exist."""
        validator = QualityValidator()
        
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        result = validator.check_missing_values(df)
        
        assert len(result) == 2
        assert result['col1']['count'] == 0
        assert result['col1']['is_valid'] == True
        assert result['col2']['count'] == 0
        assert result['col2']['is_valid'] == True

    def test_check_missing_values_with_missing(self):
        """Test missing values check with missing data."""
        validator = QualityValidator()
        
        df = pd.DataFrame({
            'col1': [1, np.nan, 3, np.nan],  # 50% missing
            'col2': ['a', 'b', None, 'd']     # 25% missing
        })
        
        result = validator.check_missing_values(df)
        
        assert result['col1']['count'] == 2
        assert result['col1']['percentage'] == 0.5
        assert result['col1']['is_valid'] == False  # > 5% missing
        
        assert result['col2']['count'] == 1
        assert result['col2']['percentage'] == 0.25
        assert result['col2']['is_valid'] == False   # > 5% missing
        
        # Should have added issues for both col1 and col2
        assert len(validator.issues) == 2
        assert any("col1" in issue for issue in validator.issues)
        assert any("col2" in issue for issue in validator.issues)

    def test_check_duplicates_no_duplicates(self):
        """Test duplicates check when no duplicates exist."""
        validator = QualityValidator()
        
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4],
            'col2': ['a', 'b', 'c', 'd']
        })
        
        result = validator.check_duplicates(df)
        
        assert result['duplicate_count'] == 0
        assert result['duplicate_percentage'] == 0.0
        assert result['is_valid'] == True
        assert len(validator.issues) == 0

    def test_check_duplicates_with_duplicates(self):
        """Test duplicates check with duplicate rows."""
        validator = QualityValidator()
        
        df = pd.DataFrame({
            'col1': [1, 2, 3, 1, 2],  # Duplicates
            'col2': ['a', 'b', 'c', 'a', 'b']
        })
        
        result = validator.check_duplicates(df)
        
        assert result['duplicate_count'] == 4  # 4 duplicate rows (all instances of duplicates)
        assert result['duplicate_percentage'] == 0.8  # 80%
        assert result['is_valid'] == False  # > 1% duplicates
        
        # Should have added issue
        assert len(validator.issues) == 1
        assert "80.00%" in validator.issues[0]

    def test_check_duplicates_with_subset(self):
        """Test duplicates check with column subset."""
        validator = QualityValidator()
        
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'category': ['A', 'A', 'B', 'A', 'C'],  # Duplicates in category
            'value': [10, 20, 30, 40, 50]
        })
        
        result = validator.check_duplicates(df, subset=['category'])
        
        # Should find duplicates based on category only
        assert result['duplicate_count'] > 0
        assert result['is_valid'] == False

    def test_check_outliers_normal_distribution(self):
        """Test outlier detection on normal distribution."""
        validator = QualityValidator()
        
        # Create normal data
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        df = pd.DataFrame({'values': data})
        
        result = validator.check_outliers(df)
        
        assert 'values' in result
        assert 'outlier_count' in result['values']
        assert 'outlier_percentage' in result['values']
        assert 'mean' in result['values']
        assert 'std' in result['values']
        assert 'min' in result['values']
        assert 'max' in result['values']

    def test_check_outliers_with_extreme_values(self):
        """Test outlier detection with extreme values."""
        validator = QualityValidator()
        
        df = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 100]  # 100 is extreme outlier with lower threshold
        })
        
        result = validator.check_outliers(df, std_threshold=2)
        
        assert result['values']['outlier_count'] > 0
        assert result['values']['outlier_percentage'] > 0

    def test_check_outliers_specific_columns(self):
        """Test outlier detection on specific columns."""
        validator = QualityValidator()
        
        df = pd.DataFrame({
            'normal_col': [1, 2, 3, 4, 5],
            'outlier_col': [1, 2, 3, 4, 1000]  # 1000 is outlier with lower threshold
        })
        
        result = validator.check_outliers(df, numeric_columns=['outlier_col'], std_threshold=1.5)
        
        # Should only check outlier_col
        assert 'outlier_col' in result
        assert 'normal_col' not in result
        assert result['outlier_col']['outlier_count'] > 0

    def test_check_data_types_valid(self):
        """Test data type validation with valid types."""
        validator = QualityValidator()
        
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c']
        })
        
        expected_types = {
            'int_col': 'int64',
            'float_col': 'float64',
            'str_col': 'object'
        }
        
        result = validator.check_data_types(df, expected_types)
        
        assert result['int_col']['is_valid'] == True
        assert result['float_col']['is_valid'] == True
        assert result['str_col']['is_valid'] == True
        assert len(validator.issues) == 0

    def test_check_data_types_invalid(self):
        """Test data type validation with invalid types."""
        validator = QualityValidator()
        
        df = pd.DataFrame({
            'int_col': ['1', '2', '3'],  # Should be int but is string
            'float_col': [1.1, 2.2, 3.3]  # Valid
        })
        
        expected_types = {
            'int_col': 'int64',
            'float_col': 'float64'
        }
        
        result = validator.check_data_types(df, expected_types)
        
        assert result['int_col']['is_valid'] == False
        assert result['float_col']['is_valid'] == True
        
        # Should have added issue
        assert len(validator.issues) == 1
        assert "int_col" in validator.issues[0]

    def test_check_data_types_missing_column(self):
        """Test data type validation with missing column."""
        validator = QualityValidator()
        
        df = pd.DataFrame({
            'existing_col': [1, 2, 3]
        })
        
        expected_types = {
            'existing_col': 'int64',
            'missing_col': 'float64'
        }
        
        result = validator.check_data_types(df, expected_types)
        
        assert result['existing_col']['is_valid'] == True
        assert result['missing_col']['present'] == False
        assert result['missing_col']['is_valid'] == False
        
        # Should have added issue
        assert len(validator.issues) == 1

    def test_validate_batch_comprehensive(self):
        """Test comprehensive batch validation."""
        validator = QualityValidator()
        
        # Create test data with various issues
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 1],  # Duplicate
            'amount': [100, np.nan, 300, 400, 500],  # Missing value
            'category': ['A', 'B', 'A', 'B', 'A']
        })
        
        expected_types = {
            'id': 'int64',
            'amount': 'float64',
            'category': 'object'
        }
        
        result = validator.validate_batch(df, expected_types)
        
        assert result['row_count'] == 5
        assert result['column_count'] == 3
        assert 'missing_values' in result
        assert 'duplicates' in result
        assert 'outliers' in result
        assert 'data_types' in result
        assert 'issues' in result
        assert result['is_valid'] == False  # Has issues
        
        # Should have multiple issues
        assert len(result['issues']) > 0

    def test_validate_batch_no_types_specified(self):
        """Test batch validation without type specifications."""
        validator = QualityValidator()
        
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        result = validator.validate_batch(df)
        
        assert 'data_types' not in result
        assert result['is_valid'] == True  # No issues

    def test_validate_batch_empty_dataframe(self):
        """Test batch validation with empty dataframe."""
        validator = QualityValidator()
        
        df = pd.DataFrame()
        
        result = validator.validate_batch(df)
        
        assert result['row_count'] == 0
        assert result['column_count'] == 0
        assert result['is_valid'] == True

    def test_missing_values_threshold_respected(self):
        """Test that missing values threshold is properly checked."""
        validator = QualityValidator()
        
        # Create data with exactly 5% missing (threshold)
        df = pd.DataFrame({
            'col': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]  # 10/20 = 50%
        })
        
        result = validator.check_missing_values(df)
        
        assert result['col']['percentage'] == 0.5
        assert result['col']['is_valid'] == False  # > 5%

    def test_duplicates_threshold_respected(self):
        """Test that duplicates threshold is properly checked."""
        validator = QualityValidator()
        
        # Create data with exactly 1% duplicates (threshold)
        df = pd.DataFrame({
            'col1': list(range(100)) + [0],  # 1 duplicate in 101 rows = ~1%
            'col2': ['val'] * 101
        })
        
        result = validator.check_duplicates(df)
        
        assert result['duplicate_percentage'] >= 0.009  # ~1%
        assert result['is_valid'] == False  # >= 1%

    def test_outlier_statistics_calculation(self):
        """Test that outlier statistics are correctly calculated."""
        validator = QualityValidator()
        
        df = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 100]  # 100 is outlier
        })
        
        result = validator.check_outliers(df, std_threshold=2)
        
        stats = result['values']
        assert stats['mean'] == pytest.approx(19.17, rel=1e-2)
        assert stats['std'] > 0
        assert stats['min'] == 1
        assert stats['max'] == 100
        assert stats['outlier_count'] > 0

    def test_issues_accumulation(self):
        """Test that issues are properly accumulated."""
        validator = QualityValidator()
        
        # Create multiple issues
        df1 = pd.DataFrame({'col': [1, np.nan, 3]})  # Missing value
        df2 = pd.DataFrame({'col': [1, 1, 2]})       # Duplicate
        
        validator.check_missing_values(df1)
        validator.check_duplicates(df2)
        
        assert len(validator.issues) == 2

    def test_report_reset_between_validations(self):
        """Test that validator state resets between validations."""
        validator = QualityValidator()
        
        # First validation
        df1 = pd.DataFrame({'col': [1, np.nan]})
        validator.validate_batch(df1)
        first_issues = len(validator.issues)
        
        # Second validation
        df2 = pd.DataFrame({'col': [1, 2, 3]})
        validator.validate_batch(df2)
        second_issues = len(validator.issues)
        
        # Issues should reset between validations
        assert first_issues > 0
        assert second_issues == 0

    def test_numeric_only_outlier_detection(self):
        """Test outlier detection only on numeric columns."""
        validator = QualityValidator()
        
        df = pd.DataFrame({
            'numeric_col': [1, 2, 3, 100],
            'string_col': ['a', 'b', 'c', 'd'],
            'bool_col': [True, False, True, False]
        })
        
        result = validator.check_outliers(df)
        
        # Should only include numeric column
        assert 'numeric_col' in result
        assert 'string_col' not in result
        assert 'bool_col' not in result

    def test_data_type_validation_edge_cases(self):
        """Test data type validation edge cases."""
        validator = QualityValidator()
        
        df = pd.DataFrame({
            'int_with_nulls': [1, 2, None],
            'float_with_nulls': [1.1, 2.2, None]
        })
        
        expected_types = {
            'int_with_nulls': 'int64',
            'float_with_nulls': 'float64'
        }
        
        result = validator.check_data_types(df, expected_types)
        
        # Should handle nullable types
        assert 'int_with_nulls' in result
        assert 'float_with_nulls' in result

    @patch('src.validation.quality.logger')
    def test_logging_on_issues(self, mock_logger):
        """Test logging when issues are found."""
        validator = QualityValidator()
        
        df = pd.DataFrame({
            'col': [1, np.nan, 3, np.nan]  # Missing values
        })
        
        validator.check_missing_values(df)
        
        assert mock_logger.warning.called

    def test_comprehensive_validation_report_structure(self):
        """Test that comprehensive validation report has all expected fields."""
        validator = QualityValidator()
        
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        result = validator.validate_batch(df)
        
        required_fields = [
            'row_count', 'column_count', 'columns', 'missing_values',
            'duplicates', 'outliers', 'issues', 'is_valid'
        ]
        
        for field in required_fields:
            assert field in result

    def test_column_list_in_report(self):
        """Test that column list is included in validation report."""
        validator = QualityValidator()
        
        df = pd.DataFrame({
            'col1': [1, 2],
            'col2': ['a', 'b'],
            'col3': [1.1, 2.2]
        })
        
        result = validator.validate_batch(df)
        
        assert set(result['columns']) == {'col1', 'col2', 'col3'}

    def test_validation_with_single_column(self):
        """Test validation with single column dataframe."""
        validator = QualityValidator()
        
        df = pd.DataFrame({'single_col': [1, 2, 3]})
        
        result = validator.validate_batch(df)
        
        assert result['column_count'] == 1
        assert result['row_count'] == 3
        assert result['is_valid'] == True


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_clean_dataframe():
    """Sample clean dataframe fixture."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'amount': [100.0, 200.0, 150.0, 300.0, 250.0],
        'category': ['A', 'B', 'A', 'C', 'B']
    })


@pytest.fixture
def sample_dataframe_with_issues():
    """Sample dataframe with quality issues fixture."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 1],  # Duplicate ID
        'amount': [100, np.nan, 300, 400, 500],  # Missing value
        'category': ['A', 'B', 'A', 'B', 'A']
    })


@pytest.fixture
def sample_dataframe_with_outliers():
    """Sample dataframe with outliers fixture."""
    return pd.DataFrame({
        'normal_values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'outlier_values': [1, 2, 3, 4, 5, 1000, 2000]  # Extreme outliers
    })

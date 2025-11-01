"""
Unit tests for Schema Validation
Tests field presence, data types, and business rules validation
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from src.validation.schema import ProductionSchemaValidator, SchemaValidator


@pytest.mark.unit
class TestSchemaValidator:
    """Test suite for SchemaValidator class."""

    def test_initialization(self):
        """Test validator initialization."""
        validator = SchemaValidator()

        assert validator.schema_type == "production"
        assert validator.schema is None
        assert validator.validation_errors == []
        assert validator.validation_warnings == []

    def test_get_schema_production(self):
        """Test getting production schema."""
        validator = SchemaValidator("production")
        schema = validator.get_schema()

        assert isinstance(schema, ProductionSchemaValidator)
        assert schema.schema_name == "production"

    def test_get_schema_invalid_type(self):
        """Test getting invalid schema type."""
        validator = SchemaValidator("invalid")

        with pytest.raises(ValueError):
            validator.get_schema("invalid")

    def test_validate_batch_success(self):
        """Test successful batch validation."""
        validator = SchemaValidator()

        # Create valid data
        df = pd.DataFrame(
            {
                "transaction_id": ["txn_001", "txn_002"],
                "customer_id": ["cust_1", "cust_2"],
                "merchant_id": ["merc_1", "merc_2"],
                "amount": [100.0, 200.0],
                "currency": ["USD", "EUR"],
                "time": ["2025-01-15T10:00:00Z", "2025-01-15T11:00:00Z"],
                "customer_zip": ["12345", "67890"],
                "merchant_zip": ["11111", "22222"],
                "customer_country": ["US", "CA"],
                "merchant_country": ["US", "CA"],
            }
        )

        is_valid, report = validator.validate_batch(df)

        assert is_valid is True
        assert report["is_valid"] is True
        assert report["total_rows"] == 2
        assert report["valid_rows"] == 2
        assert report["invalid_rows"] == 0
        assert len(report["field_errors"]) == 0
        assert len(report["business_errors"]) == 0

    def test_validate_batch_missing_required_fields(self):
        """Test batch validation with missing required fields."""
        validator = SchemaValidator()

        # Missing required fields
        df = pd.DataFrame({"amount": [100.0, 200.0], "currency": ["USD", "EUR"]})

        is_valid, report = validator.validate_batch(df)

        assert is_valid is False
        assert report["is_valid"] is False
        assert report["valid_rows"] == 0
        assert report["invalid_rows"] == 2
        assert len(report["field_errors"]) > 0
        assert "transaction_id" in report["field_errors"]

    def test_validate_batch_type_errors(self):
        """Test batch validation with type errors."""
        validator = SchemaValidator()

        df = pd.DataFrame(
            {
                "transaction_id": ["txn_001", "txn_002"],
                "customer_id": ["cust_1", "cust_2"],
                "merchant_id": ["merc_1", "merc_2"],
                "amount": ["100.0", "200.0"],  # Should be float, not string
                "currency": ["USD", "EUR"],
                "time": ["2025-01-15T10:00:00Z", "2025-01-15T11:00:00Z"],
                "customer_zip": ["12345", "67890"],
                "merchant_zip": ["11111", "22222"],
                "customer_country": ["US", "CA"],
                "merchant_country": ["US", "CA"],
            }
        )

        is_valid, report = validator.validate_batch(df)

        assert is_valid is False
        assert "type_errors" in report
        assert "amount" in report["type_errors"]

    def test_validate_batch_business_rule_violations(self):
        """Test batch validation with business rule violations."""
        validator = SchemaValidator()

        df = pd.DataFrame(
            {
                "transaction_id": ["", "txn_002"],  # Empty transaction_id
                "customer_id": ["cust_1", "cust_2"],
                "merchant_id": ["merc_1", "merc_2"],
                "amount": [-100.0, 200.0],  # Negative amount
                "currency": ["USD", "EUR"],
                "time": ["2025-01-15T10:00:00Z", "2025-01-15T11:00:00Z"],
                "customer_zip": ["12345", "67890"],
                "merchant_zip": ["11111", "22222"],
                "customer_country": ["US", "CA"],
                "merchant_country": ["US", "CA"],
            }
        )

        is_valid, report = validator.validate_batch(df)

        assert is_valid is False
        assert len(report["business_errors"]) > 0

    def test_validate_batch_empty_dataframe(self):
        """Test batch validation with empty dataframe."""
        validator = SchemaValidator()

        df = pd.DataFrame()

        is_valid, report = validator.validate_batch(df)

        assert is_valid is False
        assert report["total_rows"] == 0
        assert report["valid_rows"] == 0

    def test_get_info(self):
        """Test getting validator information."""
        validator = SchemaValidator()

        info = validator.get_info()

        assert "validator_type" in info
        assert "schema_type" in info
        assert "description" in info
        assert info["schema_type"] == "production"

    @patch("src.validation.schema.logger")
    def test_logging_on_validation_errors(self, mock_logger):
        """Test logging on validation errors."""
        validator = SchemaValidator()

        df = pd.DataFrame()  # Empty dataframe will cause errors

        validator.validate_batch(df)

        assert mock_logger.error.called


@pytest.mark.unit
class TestProductionSchemaValidator:
    """Test suite for ProductionSchemaValidator class."""

    def test_schema_properties(self):
        """Test schema property access."""
        schema = ProductionSchemaValidator()

        assert schema.schema_name == "production"
        assert len(schema.required_fields) > 0
        assert len(schema.optional_fields) > 0
        assert len(schema.field_types) > 0

        # Check required fields
        required = [
            "transaction_id",
            "customer_id",
            "merchant_id",
            "amount",
            "currency",
            "time",
        ]
        for field in required:
            assert field in schema.required_fields

    def test_validate_fields_success(self):
        """Test successful field validation."""
        schema = ProductionSchemaValidator()

        df = pd.DataFrame(
            {
                "transaction_id": ["txn_001"],
                "customer_id": ["cust_1"],
                "merchant_id": ["merc_1"],
                "amount": [100.0],
                "currency": ["USD"],
                "time": ["2025-01-15T10:00:00Z"],
                "customer_zip": ["12345"],
                "merchant_zip": ["11111"],
                "customer_country": ["US"],
                "merchant_country": ["US"],
            }
        )

        is_valid, missing = schema.validate_fields(df)

        assert is_valid is True
        assert len(missing) == 0

    def test_validate_fields_missing_required(self):
        """Test field validation with missing required fields."""
        schema = ProductionSchemaValidator()

        df = pd.DataFrame({"amount": [100.0], "currency": ["USD"]})

        is_valid, missing = schema.validate_fields(df)

        assert is_valid is False
        assert len(missing) > 0
        assert "transaction_id" in missing

    def test_validate_types_success(self):
        """Test successful type validation."""
        schema = ProductionSchemaValidator()

        df = pd.DataFrame(
            {
                "transaction_id": ["txn_001"],
                "customer_id": ["cust_1"],
                "merchant_id": ["merc_1"],
                "amount": [100.0],
                "currency": ["USD"],
                "time": ["2025-01-15T10:00:00Z"],
                "customer_zip": ["12345"],
                "merchant_zip": ["11111"],
                "customer_country": ["US"],
                "merchant_country": ["US"],
                "device_id": ["dev_001"],
                "session_id": ["sess_001"],
                "ip_address": ["192.168.1.1"],
                "mcc": [1234],
                "transaction_type": ["purchase"],
                "is_disputed": [False],
            }
        )

        is_valid, type_errors = schema.validate_types(df)

        assert is_valid is True
        assert len(type_errors) == 0

    def test_validate_types_with_errors(self):
        """Test type validation with type errors."""
        schema = ProductionSchemaValidator()

        df = pd.DataFrame(
            {
                "transaction_id": ["txn_001"],
                "customer_id": ["cust_1"],
                "merchant_id": ["merc_1"],
                "amount": ["100.0"],  # String instead of float
                "currency": ["USD"],
                "time": ["2025-01-15T10:00:00Z"],
                "customer_zip": ["12345"],
                "merchant_zip": ["11111"],
                "customer_country": ["US"],
                "merchant_country": ["US"],
                "mcc": ["1234"],  # String instead of int
                "is_disputed": ["True"],  # String instead of bool
            }
        )

        is_valid, type_errors = schema.validate_types(df)

        assert is_valid is False
        assert len(type_errors) > 0
        assert "amount" in type_errors
        assert "mcc" in type_errors
        assert "is_disputed" in type_errors

    def test_validate_business_rules_success(self):
        """Test successful business rules validation."""
        schema = ProductionSchemaValidator()

        df = pd.DataFrame(
            {
                "transaction_id": ["txn_001", "txn_002"],
                "customer_id": ["cust_1", "cust_2"],
                "merchant_id": ["merc_1", "merc_2"],
                "amount": [100.0, 200.0],
                "currency": ["USD", "EUR"],
                "time": ["2025-01-15T10:00:00Z", "2025-01-15T11:00:00Z"],
                "customer_zip": ["12345", "67890"],
                "merchant_zip": ["11111", "22222"],
                "customer_country": ["US", "CA"],
                "merchant_country": ["US", "CA"],
            }
        )

        is_valid, errors = schema.validate_business_rules(df)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_business_rules_negative_amount(self):
        """Test business rules validation with negative amount."""
        schema = ProductionSchemaValidator()

        df = pd.DataFrame(
            {
                "transaction_id": ["txn_001"],
                "customer_id": ["cust_1"],
                "merchant_id": ["merc_1"],
                "amount": [-100.0],  # Negative amount
                "currency": ["USD"],
                "time": ["2025-01-15T10:00:00Z"],
                "customer_zip": ["12345"],
                "merchant_zip": ["11111"],
                "customer_country": ["US"],
                "merchant_country": ["US"],
            }
        )

        is_valid, errors = schema.validate_business_rules(df)

        assert is_valid is False
        assert len(errors) > 0
        assert "non-negative" in errors[0].lower()

    def test_validate_business_rules_invalid_currency(self):
        """Test business rules validation with invalid currency."""
        schema = ProductionSchemaValidator()

        df = pd.DataFrame(
            {
                "transaction_id": ["txn_001"],
                "customer_id": ["cust_1"],
                "merchant_id": ["merc_1"],
                "amount": [100.0],
                "currency": ["USDD"],  # 4 characters instead of 3
                "time": ["2025-01-15T10:00:00Z"],
                "customer_zip": ["12345"],
                "merchant_zip": ["11111"],
                "customer_country": ["US"],
                "merchant_country": ["US"],
            }
        )

        is_valid, errors = schema.validate_business_rules(df)

        assert is_valid is False
        assert len(errors) > 0
        assert "currency" in errors[0].lower()

    def test_validate_business_rules_empty_transaction_id(self):
        """Test business rules validation with empty transaction_id."""
        schema = ProductionSchemaValidator()

        df = pd.DataFrame(
            {
                "transaction_id": [""],  # Empty
                "customer_id": ["cust_1"],
                "merchant_id": ["merc_1"],
                "amount": [100.0],
                "currency": ["USD"],
                "time": ["2025-01-15T10:00:00Z"],
                "customer_zip": ["12345"],
                "merchant_zip": ["11111"],
                "customer_country": ["US"],
                "merchant_country": ["US"],
            }
        )

        is_valid, errors = schema.validate_business_rules(df)

        assert is_valid is False
        assert len(errors) > 0
        assert "empty" in errors[0].lower()

    def test_validate_business_rules_missing_required_values(self):
        """Test business rules validation with missing required values."""
        schema = ProductionSchemaValidator()

        df = pd.DataFrame(
            {
                "transaction_id": ["txn_001"],
                "customer_id": ["cust_1"],
                "merchant_id": ["merc_1"],
                "amount": [np.nan],  # Missing amount
                "currency": ["USD"],
                "time": ["2025-01-15T10:00:00Z"],
                "customer_zip": ["12345"],
                "merchant_zip": ["11111"],
                "customer_country": ["US"],
                "merchant_country": ["US"],
            }
        )

        is_valid, errors = schema.validate_business_rules(df)

        assert is_valid is False
        assert len(errors) > 0
        assert "missing" in errors[0].lower()

    def test_optional_fields_handling(self):
        """Test that optional fields are handled correctly."""
        schema = ProductionSchemaValidator()

        df = pd.DataFrame(
            {
                "transaction_id": ["txn_001"],
                "customer_id": ["cust_1"],
                "merchant_id": ["merc_1"],
                "amount": [100.0],
                "currency": ["USD"],
                "time": ["2025-01-15T10:00:00Z"],
                "customer_zip": ["12345"],
                "merchant_zip": ["11111"],
                "customer_country": ["US"],
                "merchant_country": ["US"],
                # Optional fields
                "device_id": ["dev_001"],
                "session_id": [None],  # Optional can be null
                "mcc": [1234],
            }
        )

        is_valid, errors = schema.validate_business_rules(df)

        # Should still be valid even with missing optional fields
        assert is_valid is True

    def test_field_types_completeness(self):
        """Test that all fields have type definitions."""
        schema = ProductionSchemaValidator()

        # All required + optional fields should have types
        all_fields = schema.required_fields + schema.optional_fields

        for field in all_fields:
            assert field in schema.field_types

    def test_schema_validator_integration(self):
        """Test integration between SchemaValidator and ProductionSchemaValidator."""
        validator = SchemaValidator()

        df = pd.DataFrame(
            {
                "transaction_id": ["txn_001"],
                "customer_id": ["cust_1"],
                "merchant_id": ["merc_1"],
                "amount": [100.0],
                "currency": ["USD"],
                "time": ["2025-01-15T10:00:00Z"],
                "customer_zip": ["12345"],
                "merchant_zip": ["11111"],
                "customer_country": ["US"],
                "merchant_country": ["US"],
            }
        )

        is_valid, report = validator.validate_batch(df)

        assert is_valid is True
        assert report["schema_type"] == "production"

    def test_validation_error_accumulation(self):
        """Test that validation errors are properly accumulated."""
        validator = SchemaValidator()

        # DataFrame missing multiple required fields
        df = pd.DataFrame({"amount": [100.0]})

        is_valid, report = validator.validate_batch(df)

        assert is_valid is False
        assert len(report["field_errors"]) > 1  # Multiple missing fields

    def test_validation_with_extra_columns(self):
        """Test validation with extra columns not in schema."""
        validator = SchemaValidator()

        df = pd.DataFrame(
            {
                "transaction_id": ["txn_001"],
                "customer_id": ["cust_1"],
                "merchant_id": ["merc_1"],
                "amount": [100.0],
                "currency": ["USD"],
                "time": ["2025-01-15T10:00:00Z"],
                "customer_zip": ["12345"],
                "merchant_zip": ["11111"],
                "customer_country": ["US"],
                "merchant_country": ["US"],
                "extra_column": ["extra_value"],  # Extra column
            }
        )

        is_valid, report = validator.validate_batch(df)

        # Should still be valid - extra columns are allowed
        assert is_valid is True

    def test_edge_case_single_row_validation(self):
        """Test validation with single row."""
        validator = SchemaValidator()

        df = pd.DataFrame(
            {
                "transaction_id": ["txn_001"],
                "customer_id": ["cust_1"],
                "merchant_id": ["merc_1"],
                "amount": [100.0],
                "currency": ["USD"],
                "time": ["2025-01-15T10:00:00Z"],
                "customer_zip": ["12345"],
                "merchant_zip": ["11111"],
                "customer_country": ["US"],
                "merchant_country": ["US"],
            }
        )

        is_valid, report = validator.validate_batch(df)

        assert is_valid is True
        assert report["total_rows"] == 1
        assert report["valid_rows"] == 1

    def test_edge_case_maximum_rows_validation(self):
        """Test validation with many rows."""
        validator = SchemaValidator()

        # Create large dataframe
        n_rows = 1000
        df = pd.DataFrame(
            {
                "transaction_id": [f"txn_{i}" for i in range(n_rows)],
                "customer_id": [f"cust_{i}" for i in range(n_rows)],
                "merchant_id": [f"merc_{i}" for i in range(n_rows)],
                "amount": [100.0 + i for i in range(n_rows)],
                "currency": ["USD"] * n_rows,
                "time": ["2025-01-15T10:00:00Z"] * n_rows,
                "customer_zip": ["12345"] * n_rows,
                "merchant_zip": ["11111"] * n_rows,
                "customer_country": ["US"] * n_rows,
                "merchant_country": ["US"] * n_rows,
            }
        )

        is_valid, report = validator.validate_batch(df)

        assert is_valid is True
        assert report["total_rows"] == n_rows
        assert report["valid_rows"] == n_rows


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_valid_transaction():
    """Sample valid transaction fixture."""
    return pd.DataFrame(
        {
            "transaction_id": ["txn_001"],
            "customer_id": ["cust_1"],
            "merchant_id": ["merc_1"],
            "amount": [100.0],
            "currency": ["USD"],
            "time": ["2025-01-15T10:00:00Z"],
            "customer_zip": ["12345"],
            "merchant_zip": ["11111"],
            "customer_country": ["US"],
            "merchant_country": ["US"],
        }
    )


@pytest.fixture
def sample_invalid_transaction():
    """Sample invalid transaction fixture."""
    return pd.DataFrame(
        {
            "transaction_id": [""],  # Empty
            "customer_id": ["cust_1"],
            "merchant_id": ["merc_1"],
            "amount": [-100.0],  # Negative
            "currency": ["USDD"],  # Invalid currency
            "time": ["2025-01-15T10:00:00Z"],
            "customer_zip": ["12345"],
            "merchant_zip": ["11111"],
            "customer_country": ["US"],
            "merchant_country": ["US"],
        }
    )


@pytest.fixture
def sample_batch_transactions():
    """Sample batch of transactions fixture."""
    return pd.DataFrame(
        {
            "transaction_id": ["txn_001", "txn_002", "txn_003"],
            "customer_id": ["cust_1", "cust_2", "cust_3"],
            "merchant_id": ["merc_1", "merc_2", "merc_3"],
            "amount": [100.0, 200.0, 300.0],
            "currency": ["USD", "EUR", "GBP"],
            "time": [
                "2025-01-15T10:00:00Z",
                "2025-01-15T11:00:00Z",
                "2025-01-15T12:00:00Z",
            ],
            "customer_zip": ["12345", "67890", "11111"],
            "merchant_zip": ["11111", "22222", "33333"],
            "customer_country": ["US", "CA", "UK"],
            "merchant_country": ["US", "CA", "UK"],
        }
    )

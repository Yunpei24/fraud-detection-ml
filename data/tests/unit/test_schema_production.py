"""
Unit tests for production schema validation

Tests the production schema validator for Event Hub/Kafka transactions.
"""

import pytest
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.validation.schema import SchemaValidator, ProductionSchemaValidator


class TestSchemaValidator:
    """Tests for SchemaValidator"""

    def test_initialize_validator(self):
        """Test initializing SchemaValidator"""
        validator = SchemaValidator()
        assert validator is not None
        assert validator.schema_type == 'production'

    def test_validate_batch_production_valid(self):
        """Test batch validation with valid production data"""
        validator = SchemaValidator()
        
        # Create valid production data
        df = pd.DataFrame({
            'transaction_id': ['TXN001', 'TXN002', 'TXN003'],
            'customer_id': ['CUST001', 'CUST002', 'CUST003'],
            'merchant_id': ['MRCH001', 'MRCH002', 'MRCH001'],
            'amount': [100.50, 250.00, 75.25],
            'currency': ['USD', 'USD', 'EUR'],
            'transaction_time': ['2025-10-19 10:30:00', '2025-10-19 11:45:00', '2025-10-19 12:00:00'],
            'customer_zip': ['12345', '54321', '11111'],
            'merchant_zip': ['99999', '88888', '77777'],
            'customer_country': ['US', 'US', 'DE'],
            'merchant_country': ['US', 'US', 'DE'],
        })
        
        is_valid, report = validator.validate_batch(df, schema_type='production')
        assert is_valid is True
        assert report['schema_type'] == 'production'
        assert report['valid_rows'] == 3

    def test_validate_batch_missing_required_fields(self):
        """Test batch validation with missing required fields"""
        validator = SchemaValidator()
        
        # Missing merchant_id and other required fields
        df = pd.DataFrame({
            'transaction_id': ['TXN001'],
            'customer_id': ['CUST001'],
            'amount': [100.50],
            'currency': ['USD'],
            # Missing: merchant_id, transaction_time, customer_zip, merchant_zip, etc.
        })
        
        is_valid, report = validator.validate_batch(df, schema_type='production')
        assert is_valid is False
        assert len(report['field_errors']) > 0

    def test_validate_batch_negative_amount(self):
        """Test batch validation with negative amount"""
        validator = SchemaValidator()
        
        df = pd.DataFrame({
            'transaction_id': ['TXN001'],
            'customer_id': ['CUST001'],
            'merchant_id': ['MRCH001'],
            'amount': [-100.50],  # Invalid: negative
            'currency': ['USD'],
            'transaction_time': ['2025-10-19 10:30:00'],
            'customer_zip': ['12345'],
            'merchant_zip': ['99999'],
            'customer_country': ['US'],
            'merchant_country': ['US'],
        })
        
        is_valid, report = validator.validate_batch(df, schema_type='production')
        assert is_valid is False
        assert len(report['business_errors']) > 0

    def test_validate_batch_invalid_currency(self):
        """Test batch validation with invalid currency code"""
        validator = SchemaValidator()
        
        df = pd.DataFrame({
            'transaction_id': ['TXN001'],
            'customer_id': ['CUST001'],
            'merchant_id': ['MRCH001'],
            'amount': [100.50],
            'currency': ['INVALID'],  # Invalid: should be 3-letter code
            'transaction_time': ['2025-10-19 10:30:00'],
            'customer_zip': ['12345'],
            'merchant_zip': ['99999'],
            'customer_country': ['US'],
            'merchant_country': ['US'],
        })
        
        is_valid, report = validator.validate_batch(df, schema_type='production')
        assert is_valid is False
        assert len(report['business_errors']) > 0

    def test_validate_batch_invalid_schema_type(self):
        """Test with invalid schema type"""
        validator = SchemaValidator()
        
        df = pd.DataFrame({'transaction_id': ['TXN001']})
        
        # validate_batch catches exceptions and returns them in report
        is_valid, report = validator.validate_batch(df, schema_type='invalid_schema')
        assert is_valid is False
        assert 'errors' in report

    def test_get_schema(self):
        """Test getting production schema"""
        validator = SchemaValidator()
        schema = validator.get_schema('production')
        assert isinstance(schema, ProductionSchemaValidator)
        assert schema.schema_name == 'production'


class TestProductionSchemaValidator:
    """Tests for ProductionSchemaValidator"""

    def test_schema_properties(self):
        """Test ProductionSchemaValidator properties"""
        schema = ProductionSchemaValidator()
        assert schema.schema_name == 'production'
        assert 'transaction_id' in schema.required_fields
        assert 'customer_id' in schema.required_fields
        assert 'merchant_id' in schema.required_fields
        assert 'amount' in schema.required_fields
        assert len(schema.required_fields) >= 10

    def test_validate_fields_present(self):
        """Test field validation when all required fields present"""
        schema = ProductionSchemaValidator()
        
        df = pd.DataFrame({
            'transaction_id': ['TXN001'],
            'customer_id': ['CUST001'],
            'merchant_id': ['MRCH001'],
            'amount': [100.50],
            'currency': ['USD'],
            'transaction_time': ['2025-10-19 10:30:00'],
            'customer_zip': ['12345'],
            'merchant_zip': ['99999'],
            'customer_country': ['US'],
            'merchant_country': ['US'],
        })
        
        is_valid, missing = schema.validate_fields(df)
        assert is_valid is True
        assert len(missing) == 0

    def test_validate_fields_missing(self):
        """Test field validation when required fields missing"""
        schema = ProductionSchemaValidator()
        
        df = pd.DataFrame({
            'transaction_id': ['TXN001'],
            # Missing all other required fields
        })
        
        is_valid, missing = schema.validate_fields(df)
        assert is_valid is False
        assert len(missing) > 0
        assert 'customer_id' in missing

    def test_validate_business_rules_valid(self):
        """Test business rule validation with valid data"""
        schema = ProductionSchemaValidator()
        
        df = pd.DataFrame({
            'transaction_id': ['TXN001', 'TXN002'],
            'customer_id': ['CUST001', 'CUST002'],
            'merchant_id': ['MRCH001', 'MRCH002'],
            'amount': [100.50, 250.00],
            'currency': ['USD', 'EUR'],
            'transaction_time': ['2025-10-19 10:30:00', '2025-10-19 11:00:00'],
            'customer_zip': ['12345', '54321'],
            'merchant_zip': ['99999', '88888'],
            'customer_country': ['US', 'DE'],
            'merchant_country': ['US', 'DE'],
        })
        
        is_valid, errors = schema.validate_business_rules(df)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_business_rules_negative_amount(self):
        """Test business rule validation with negative amount"""
        schema = ProductionSchemaValidator()
        
        df = pd.DataFrame({
            'transaction_id': ['TXN001'],
            'customer_id': ['CUST001'],
            'merchant_id': ['MRCH001'],
            'amount': [-50.00],  # Invalid
            'currency': ['USD'],
            'transaction_time': ['2025-10-19 10:30:00'],
            'customer_zip': ['12345'],
            'merchant_zip': ['99999'],
            'customer_country': ['US'],
            'merchant_country': ['US'],
        })
        
        is_valid, errors = schema.validate_business_rules(df)
        assert is_valid is False
        assert any('negative' in str(e).lower() for e in errors)

    def test_validate_business_rules_empty_transaction_id(self):
        """Test business rule validation with empty transaction_id"""
        schema = ProductionSchemaValidator()
        
        df = pd.DataFrame({
            'transaction_id': [''],  # Invalid: empty
            'customer_id': ['CUST001'],
            'merchant_id': ['MRCH001'],
            'amount': [100.50],
            'currency': ['USD'],
            'transaction_time': ['2025-10-19 10:30:00'],
            'customer_zip': ['12345'],
            'merchant_zip': ['99999'],
            'customer_country': ['US'],
            'merchant_country': ['US'],
        })
        
        is_valid, errors = schema.validate_business_rules(df)
        assert is_valid is False

    def test_validate_business_rules_missing_value(self):
        """Test business rule validation with missing values"""
        schema = ProductionSchemaValidator()
        
        df = pd.DataFrame({
            'transaction_id': ['TXN001', None],  # NaN value
            'customer_id': ['CUST001', 'CUST002'],
            'merchant_id': ['MRCH001', 'MRCH002'],
            'amount': [100.50, 250.00],
            'currency': ['USD', 'EUR'],
            'transaction_time': ['2025-10-19 10:30:00', '2025-10-19 11:00:00'],
            'customer_zip': ['12345', '54321'],
            'merchant_zip': ['99999', '88888'],
            'customer_country': ['US', 'DE'],
            'merchant_country': ['US', 'DE'],
        })
        
        is_valid, errors = schema.validate_business_rules(df)
        assert is_valid is False
        assert any('missing' in str(e).lower() for e in errors)

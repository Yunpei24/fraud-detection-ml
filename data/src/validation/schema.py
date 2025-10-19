"""
Schema validation for transaction data.

Validates incoming transaction events from Event Hub/Kafka against
the production transaction schema. This is the real-world schema
used in the fraud detection system.
"""

from typing import Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class SchemaValidator:
    """
    Validates transaction data against production schema.
    
    Validates events from Event Hub/Kafka for:
    - Required fields presence
    - Data type correctness
    - Business rule compliance
    - Data quality checks
    
    Example:
        validator = SchemaValidator()
        is_valid, report = validator.validate_batch(df, schema_type='production')
    """
    
    AVAILABLE_SCHEMAS = {
        'production': None,  # Will be dynamically loaded
    }
    
    def __init__(self, schema_type: str = 'production'):
        """
        Initialize schema validator.
        
        Args:
            schema_type (str): Schema type to use. Default: 'production'
        """
        self.schema_type = schema_type
        self.schema: Optional[ProductionSchemaValidator] = None
        self.validation_errors = []
        self.validation_warnings = []
    
    def get_schema(self, schema_type: Optional[str] = None) -> 'ProductionSchemaValidator':
        """
        Get schema validator for the specified type.
        
        For now, only 'production' schema is supported.
        All incoming events from Event Hub/Kafka are validated
        against the production transaction schema.
        
        Args:
            schema_type (str, optional): Schema type to use. If None, uses self.schema_type.
            
        Returns:
            BaseSchema: Schema validator instance
            
        Raises:
            ValueError: If schema type is not supported
        """
        schema_type = schema_type or self.schema_type
        
        if schema_type != 'production':
            raise ValueError(
                f"Only 'production' schema is supported. "
                f"Got: {schema_type}. "
                f"Use BaseSchema subclass for custom schemas."
            )
        
        # Return a basic validator for production schema
        # This can be extended with a ProductionSchema class
        return ProductionSchemaValidator()
    
    def validate_batch(self, df: pd.DataFrame, schema_type: str = 'production') -> tuple[bool, dict]:
        """
        Validate a batch of transaction data.
        
        Called for events from Event Hub/Kafka batch processing.
        
        Args:
            df (pd.DataFrame): Transaction data to validate
            schema_type (str): Schema type. Default: 'production'
            
        Returns:
            tuple[bool, dict]: (is_valid, validation_report)
        """
        self.validation_errors = []
        self.validation_warnings = []
        
        report = {
            "schema_type": schema_type,
            "total_rows": len(df),
            "valid_rows": 0,
            "invalid_rows": 0,
            "field_errors": [],
            "type_errors": {},
            "business_errors": [],
            "is_valid": False
        }
        
        try:
            # Get schema validator (currently only 'production' supported)
            schema = self.get_schema(schema_type)
            
            # Check required fields
            fields_valid, missing = schema.validate_fields(df)
            if not fields_valid:
                report["field_errors"] = missing
                report["invalid_rows"] = len(df)
                return False, report
            
            # Check field types
            types_valid, type_errors = schema.validate_types(df)
            if type_errors:
                report["type_errors"] = type_errors
            
            # Check business rules
            business_valid, business_errors = schema.validate_business_rules(df)
            if business_errors:
                report["business_errors"] = business_errors
            
            # Determine overall validity
            report["valid_rows"] = len(df) if (fields_valid and types_valid and business_valid) else 0
            report["invalid_rows"] = 0 if (fields_valid and types_valid and business_valid) else len(df)
            report["is_valid"] = fields_valid and types_valid and business_valid
            
            return report["is_valid"], report
            
        except Exception as e:
            report["is_valid"] = False
            report["errors"] = [str(e)]
            self.validation_errors.append(str(e))
            logger.error(f"Validation error: {str(e)}")
            return False, report
    
    def get_info(self) -> dict:
        """Get validator information."""
        return {
            "validator_type": "ProductionSchemaValidator",
            "schema_type": "production",
            "description": "Validates transaction events from Event Hub/Kafka",
        }


class ProductionSchemaValidator:
    """
    Production schema validator for transaction events.
    
    Validates transaction events from Event Hub/Kafka against
    the production transaction schema defined in the architecture.
    
    Expected fields:
    - transaction_id: Unique identifier for the transaction
    - customer_id: Customer identifier  
    - merchant_id: Merchant identifier
    - amount: Transaction amount
    - currency: Currency code
    - transaction_time: Timestamp of transaction
    - customer_zip, merchant_zip: Location data
    - customer_country, merchant_country: Country codes
    - device_id: Device identifier (optional)
    - session_id: Session identifier (optional)
    - ip_address: IP address (optional)
    - mcc: Merchant category code (optional)
    - transaction_type: Type of transaction (optional)
    - is_disputed: Whether transaction is disputed (optional)
    - source_system: Source system identifier
    """
    
    @property
    def schema_name(self) -> str:
        """Name of this schema."""
        return "production"
    
    @property
    def required_fields(self) -> list:
        """Fields that must be present."""
        return [
            'transaction_id', 'customer_id', 'merchant_id',
            'amount', 'currency', 'transaction_time',
            'customer_zip', 'merchant_zip',
            'customer_country', 'merchant_country'
        ]
    
    @property
    def optional_fields(self) -> list:
        """Optional fields."""
        return [
            'device_id', 'session_id', 'ip_address',
            'mcc', 'transaction_type', 'is_disputed',
            'source_system', 'ingestion_timestamp'
        ]
    
    @property
    def field_types(self) -> dict:
        """Expected data types."""
        return {
            'transaction_id': str,
            'customer_id': str,
            'merchant_id': str,
            'amount': float,
            'currency': str,
            'transaction_time': str,  # Timestamp as string (will be parsed)
            'customer_zip': str,
            'merchant_zip': str,
            'customer_country': str,
            'merchant_country': str,
            'device_id': str,
            'session_id': str,
            'ip_address': str,
            'mcc': int,
            'transaction_type': str,
            'is_disputed': bool,
        }
    
    def validate_fields(self, df: pd.DataFrame) -> tuple[bool, list]:
        """
        Validate that all required fields are present.
        
        Args:
            df (pd.DataFrame): Data to validate
            
        Returns:
            tuple[bool, list]: (is_valid, list_of_missing_fields)
        """
        missing_fields = []
        for field in self.required_fields:
            if field not in df.columns:
                missing_fields.append(field)
        
        return len(missing_fields) == 0, missing_fields
    
    def validate_types(self, df: pd.DataFrame) -> tuple[bool, dict]:
        """
        Validate that field types are correct.
        
        Args:
            df (pd.DataFrame): Data to validate
            
        Returns:
            tuple[bool, dict]: (is_valid, dict_of_type_errors)
        """
        type_errors = {}
        
        for field, expected_type in self.field_types.items():
            if field not in df.columns:
                continue
            
            # Basic type check
            try:
                if expected_type == str:
                    if not all(isinstance(x, (str, type(None))) for x in df[field]):
                        type_errors[field] = f"Expected string, got mixed types"
                elif expected_type == int:
                    if not all(isinstance(x, (int, type(None))) for x in df[field]):
                        type_errors[field] = f"Expected int, got mixed types"
                elif expected_type == float:
                    if not all(isinstance(x, (int, float, type(None))) for x in df[field]):
                        type_errors[field] = f"Expected float, got mixed types"
                elif expected_type == bool:
                    if not all(isinstance(x, (bool, type(None))) for x in df[field]):
                        type_errors[field] = f"Expected bool, got mixed types"
            except Exception as e:
                type_errors[field] = str(e)
        
        return len(type_errors) == 0, type_errors
    
    def validate_business_rules(self, df: pd.DataFrame) -> tuple[bool, list]:
        """
        Validate production-specific business rules.
        
        Args:
            df (pd.DataFrame): Data to validate
            
        Returns:
            tuple[bool, list]: (is_valid, list_of_errors)
        """
        errors = []
        
        # Check for missing values in required fields
        for field in self.required_fields:
            if field in df.columns and df[field].isna().any():
                null_count = df[field].isna().sum()
                errors.append(f"Field '{field}' has {null_count} missing values")
        
        # Validate amount (should be positive)
        if 'amount' in df.columns:
            if (df['amount'] < 0).any():
                errors.append("Amount values should be non-negative")
        
        # Validate currency (basic check - should be 3-letter code)
        if 'currency' in df.columns:
            invalid_currencies = df[df['currency'].str.len() != 3]
            if len(invalid_currencies) > 0:
                errors.append(f"Found {len(invalid_currencies)} invalid currency codes (expected 3-letter codes)")
        
        # Validate transaction_id (should not be empty)
        if 'transaction_id' in df.columns:
            if (df['transaction_id'] == '').any() or df['transaction_id'].isna().any():
                errors.append("transaction_id cannot be empty")
        
        return len(errors) == 0, errors

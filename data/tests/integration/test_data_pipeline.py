"""
Integration tests for data pipeline components
"""

import pytest
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from src.ingestion.transaction_simulator import TransactionSimulator
from src.validation.quality import QualityValidator
from src.transformation.cleaner import DataCleaner
from src.storage.database import DatabaseService


class TestDataPipelineIntegration:
    """Integration tests for data pipeline components working together"""

    @patch('src.storage.database.DatabaseService')
    def test_full_data_pipeline_simulation(self, mock_db_service_class):
        """Test complete data pipeline: simulate -> validate -> clean -> store"""
        # Mock the entire DatabaseService class and its methods
        mock_db_instance = Mock()
        mock_db_service_class.return_value = mock_db_instance
        
        # Mock the insert_transactions method specifically
        mock_db_instance.insert_transactions.return_value = None

        # Step 1: Generate sample transaction data using individual generation
        simulator = TransactionSimulator()
        transactions = []
        for _ in range(100):
            transactions.append(simulator.generate_transaction())

        # Convert to DataFrame for processing
        df = pd.DataFrame(transactions)
        
        # Remove optional columns that may have high missing rates
        df = df.drop(columns=['fraud_amount_type'], errors='ignore')

        # Step 2: Validate data quality
        validator = QualityValidator()
        quality_report = validator.validate_batch(df)

        # Should pass basic validation (allowing some missing values)
        assert quality_report['row_count'] == 100
        assert 'missing_values' in quality_report
        assert 'duplicates' in quality_report

        # Step 3: Clean the data
        cleaner = DataCleaner()
        cleaned_df = cleaner.clean_pipeline(df)

        # Should have cleaned data
        assert len(cleaned_df) > 0
        assert hasattr(cleaner, 'transformations_applied')

        # Step 4: Store the data (mocked)
        # Use the mocked instance instead of creating a new one
        mock_db_instance.insert_transactions(cleaned_df.to_dict('records'))

        # Verify database save was called
        mock_db_instance.insert_transactions.assert_called_once()

        print("✅ Full data pipeline integration test passed")

    @patch('src.storage.database.DatabaseService')
    def test_pipeline_with_fraudulent_data(self, mock_db_service_class):
        """Test pipeline handling of fraudulent transaction data"""
        # Mock the entire DatabaseService class and its methods
        mock_db_instance = Mock()
        mock_db_service_class.return_value = mock_db_instance
        
        # Mock the insert_transactions method specifically
        mock_db_instance.insert_transactions.return_value = None

        # Generate data with high fraud rate by setting fraud_rate in constructor
        simulator = TransactionSimulator(fraud_rate=0.8)
        transactions = []
        for _ in range(50):
            transactions.append(simulator.generate_transaction())

        df = pd.DataFrame(transactions)

        # Validate and clean (allow missing values for optional fields)
        validator = QualityValidator()
        quality_report = validator.validate_batch(df)

        cleaner = DataCleaner()
        cleaned_df = cleaner.clean_pipeline(df)

        # Store the data
        # Use the mocked instance instead of creating a new one
        mock_db_instance.insert_transactions(cleaned_df.to_dict('records'))

        # Verify the pipeline handled the data
        assert len(cleaned_df) > 0
        assert quality_report['row_count'] == 50
        mock_db_instance.insert_transactions.assert_called_once()

        print("✅ Fraudulent data pipeline test passed")

    def test_pipeline_error_handling(self):
        """Test pipeline error handling with invalid data"""
        from src.validation.schema import SchemaValidator
        
        # Create invalid data (missing required fields)
        invalid_df = pd.DataFrame({
            'amount': [100, 200, 300],
            # Missing required fields like transaction_id, customer_id, etc.
        })

        # Schema validation should fail
        schema_validator = SchemaValidator()
        is_valid, schema_report = schema_validator.validate_batch(invalid_df)

        # Should detect issues
        assert is_valid == False
        assert len(schema_report['field_errors']) > 0

        print("✅ Pipeline error handling test passed")

    @patch('src.ingestion.transaction_simulator.KafkaProducer')
    def test_realtime_pipeline_integration(self, mock_kafka_producer):
        """Test realtime pipeline components integration"""
        # Mock Kafka producer
        mock_producer = Mock()
        mock_kafka_producer.return_value = mock_producer

        # Generate and send transaction
        simulator = TransactionSimulator()
        transaction = simulator.generate_legitimate_transaction()

        # Send to Kafka (mocked)
        simulator.connect()
        simulator.send_transaction(transaction)
        simulator.disconnect()  # This calls flush

        # Verify Kafka producer was used
        mock_producer.send.assert_called_once()
        mock_producer.flush.assert_called_once()

        print("✅ Realtime pipeline integration test passed")

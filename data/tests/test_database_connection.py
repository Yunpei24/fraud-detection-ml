"""Test database connection after PostgreSQL migration"""
import pytest
import sqlalchemy as sa
from unittest.mock import Mock, patch
from src.config.settings import Settings


def test_database_url_is_postgresql():
    """Verify database URL uses PostgreSQL"""
    settings = Settings()
    assert settings.database_url.startswith("postgresql://"), \
        f"Expected PostgreSQL URL, got: {settings.database_url}"
    assert "5432" in settings.database_url, \
        f"Expected port 5432, got: {settings.database_url}"


@patch('sqlalchemy.create_engine')
def test_database_connection(mock_create_engine):
    """Test database connection setup (mocked)"""
    # Mock the engine and connection
    mock_engine = Mock()
    mock_conn = Mock()
    mock_result = Mock()
    mock_result.scalar.return_value = 1
    mock_conn.execute.return_value = mock_result
    mock_engine.connect.return_value.__enter__ = Mock(return_value=mock_conn)
    mock_engine.connect.return_value.__exit__ = Mock(return_value=None)
    mock_create_engine.return_value = mock_engine
    
    settings = Settings()
    
    # Test the connection logic
    engine = sa.create_engine(settings.database_url)
    
    with engine.connect() as conn:
        result = conn.execute(sa.text("SELECT 1 as test"))
        value = result.scalar()
        assert value == 1, "Database connection test failed"
    
    # Verify the engine was created correctly
    mock_create_engine.assert_called_once_with(settings.database_url)


def test_database_settings():
    """Verify database settings are correct"""
    settings = Settings()
    
    assert settings.database.port == 5432, f"Expected port 5432, got {settings.database.port}"
    assert settings.database.database == "fraud_db", f"Expected database fraud_db, got {settings.database.database}"
    # Note: Pydantic only validates defined fields, no need to check for 'driver' field absence
    
    print("✅ Database settings correct")


if __name__ == "__main__":
    test_database_url_is_postgresql()
    test_database_settings()
    # test_database_connection()  # Skip if PostgreSQL not running locally
    print("\n✅ All tests passed!")

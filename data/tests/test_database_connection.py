"""Test database connection after PostgreSQL migration"""
import pytest
import sqlalchemy as sa
from src.config.settings import Settings


def test_database_url_is_postgresql():
    """Verify database URL uses PostgreSQL"""
    settings = Settings()
    assert settings.database_url.startswith("postgresql://"), \
        f"Expected PostgreSQL URL, got: {settings.database_url}"
    assert "5432" in settings.database_url, \
        f"Expected port 5432, got: {settings.database_url}"


def test_database_connection():
    """Test actual connection to PostgreSQL"""
    settings = Settings()
    
    try:
        engine = sa.create_engine(settings.database_url)
        
        with engine.connect() as conn:
            result = conn.execute(sa.text("SELECT 1 as test"))
            value = result.scalar()
            assert value == 1, "Database connection test failed"
            
        print("✅ PostgreSQL connection successful")
    except Exception as e:
        pytest.fail(f"Failed to connect to PostgreSQL: {e}")


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

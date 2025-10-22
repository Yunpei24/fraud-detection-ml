"""
Configuration settings for the data pipeline
Supports environment variables for cloud deployment
Migrated to Pydantic Settings for consistency with API, Drift, and Airflow modules
"""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class AzureSettings(BaseSettings):
    """Azure cloud configuration"""
    connection_string: str = Field(
        default="DefaultEndpointsProtocol=https;AccountName=devaccount;AccountKey=devkey;EndpointSuffix=core.windows.net",
        env="AZURE_STORAGE_CONNECTION_STRING",
        description="Azure Storage connection string"
    )
    event_hub_name: str = Field(
        default="fraud-transactions",
        env="EVENT_HUB_NAME",
        description="Azure Event Hub name for transaction streaming"
    )
    event_hub_connection_string: str = Field(
        default="Endpoint=sb://dev.servicebus.windows.net/;SharedAccessKeyName=RootManageSharedAccessKey;SharedAccessKey=devkey",
        env="EVENT_HUB_CONNECTION_STRING",
        description="Azure Event Hub connection string"
    )
    storage_account_name: str = Field(
        default="frauddetectiondl",
        env="AZURE_STORAGE_ACCOUNT",
        description="Azure Storage account name"
    )
    storage_account_key: str = Field(
        default="devkey",
        env="AZURE_STORAGE_KEY",
        description="Azure Storage account key"
    )
    data_lake_path: str = Field(
        default="/data/transactions",
        env="AZURE_DATA_LAKE_PATH",
        description="Path in Azure Data Lake for transactions"
    )

    class Config:
        env_file = ".env"
        case_sensitive = False


class DatabaseSettings(BaseSettings):
    """Database configuration"""
    server: str = Field(
        default="localhost",
        env="DB_SERVER",
        description="Database server hostname"
    )
    database: str = Field(
        default="fraud_db",
        env="DB_NAME",
        description="Database name"
    )
    username: str = Field(
        default="postgres",
        env="DB_USER",
        description="Database username"
    )
    password: str = Field(
        default="postgres",
        env="DB_PASSWORD",
        description="Database password"
    )
    port: int = Field(
        default=5432,
        env="DB_PORT",
        description="Database port (PostgreSQL default: 5432)"
    )
    pool_size: int = Field(
        default=20,
        env="DB_POOL_SIZE",
        description="SQLAlchemy connection pool size"
    )
    max_overflow: int = Field(
        default=40,
        env="DB_MAX_OVERFLOW",
        description="SQLAlchemy max overflow connections"
    )

    class Config:
        env_file = ".env"
        case_sensitive = False


class KafkaSettings(BaseSettings):
    """Kafka configuration (alternative to Event Hub)"""
    bootstrap_servers: str = Field(
        default="localhost:9092",
        env="KAFKA_BROKERS",
        description="Comma-separated list of Kafka brokers"
    )
    topic: str = Field(
        default="fraud-transactions",
        env="KAFKA_TOPIC",
        description="Kafka topic for transaction streaming"
    )
    group_id: str = Field(
        default="fraud-detection-group",
        env="KAFKA_GROUP_ID",
        description="Kafka consumer group ID"
    )
    consumer_timeout_ms: int = Field(
        default=3000,
        env="KAFKA_TIMEOUT_MS",
        description="Kafka consumer timeout in milliseconds"
    )

    @property
    def bootstrap_servers_list(self) -> list:
        """Convert comma-separated servers to list"""
        return self.bootstrap_servers.split(",")

    class Config:
        env_file = ".env"
        case_sensitive = False


class CacheSettings(BaseSettings):
    """Redis cache configuration"""
    host: str = Field(
        default="localhost",
        env="REDIS_HOST",
        description="Redis server hostname"
    )
    port: int = Field(
        default=6379,
        env="REDIS_PORT",
        description="Redis server port"
    )
    db: int = Field(
        default=0,
        env="REDIS_DB",
        description="Redis database number"
    )
    password: Optional[str] = Field(
        default=None,
        env="REDIS_PASSWORD",
        description="Redis password (optional)"
    )
    ttl_seconds: int = Field(
        default=3600,
        env="CACHE_TTL_SECONDS",
        description="Default cache TTL in seconds"
    )

    class Config:
        env_file = ".env"
        case_sensitive = False


class MonitoringSettings(BaseSettings):
    """Monitoring and observability"""
    prometheus_port: int = Field(
        default=9092,
        env="PROMETHEUS_PORT",
        description="Port for Prometheus metrics endpoint"
    )
    log_level: str = Field(
        default="INFO",
        env="LOG_LEVEL",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    enable_profiling: bool = Field(
        default=False,
        env="ENABLE_PROFILING",
        description="Enable performance profiling"
    )
    enable_data_validation: bool = Field(
        default=True,
        env="ENABLE_DATA_VALIDATION",
        description="Enable data quality validation"
    )

    class Config:
        env_file = ".env"
        case_sensitive = False


class Settings(BaseSettings):
    """
    Main settings class that loads configuration from environment variables
    Uses Pydantic for validation and type checking
    Compatible with Docker Compose and .env files
    """
    
    # Environment
    env: str = Field(
        default="development",
        env="ENV",
        description="Environment: development, staging, production"
    )
    debug: bool = Field(
        default=False,
        env="DEBUG",
        description="Enable debug mode"
    )

    # Nested settings (instantiated on access)
    azure: AzureSettings = Field(default_factory=AzureSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    kafka: KafkaSettings = Field(default_factory=KafkaSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)

    @property
    def database_url(self) -> str:
        """Construct database connection URL for SQLAlchemy"""
        return (
            f"postgresql://{self.database.username}:{self.database.password}"
            f"@{self.database.server}:{self.database.port}/{self.database.database}"
        )

    class Config:
        env_file = ".env"
        case_sensitive = False
        env_nested_delimiter = "__"  # Support KAFKA__TOPIC=fraud-tx

    def __repr__(self) -> str:
        return (
            f"Settings(env={self.env}, debug={self.debug}, "
            f"database={self.database.server}, cache={self.cache.host})"
        )


# Singleton instance
settings = Settings()

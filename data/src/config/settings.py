"""
Configuration settings for the data pipeline
Supports environment variables for cloud deployment
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class AzureSettings:
    """Azure cloud configuration"""
    connection_string: str
    event_hub_name: str
    event_hub_connection_string: str
    storage_account_name: str
    storage_account_key: str
    data_lake_path: str


@dataclass
class DatabaseSettings:
    """Database configuration"""
    driver: str  # e.g., "ODBC Driver 17 for SQL Server"
    server: str
    database: str
    username: str
    password: str
    port: int = 1433
    pool_size: int = 20
    max_overflow: int = 40


@dataclass
class KafkaSettings:
    """Kafka configuration (alternative to Event Hub)"""
    bootstrap_servers: list
    topic: str
    group_id: str
    consumer_timeout_ms: int = 3000


@dataclass
class CacheSettings:
    """Redis cache configuration"""
    host: str
    port: int
    db: int
    password: Optional[str] = None
    ttl_seconds: int = 3600


@dataclass
class MonitoringSettings:
    """Monitoring and observability"""
    prometheus_port: int
    log_level: str
    enable_profiling: bool
    enable_data_validation: bool


class Settings:
    """
    Main settings class that loads configuration from environment variables
    or uses default values
    """

    def __init__(self):
        # Environment
        self.env = os.getenv("ENV", "development")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"

        # Azure
        self.azure = AzureSettings(
            connection_string=os.getenv(
                "AZURE_STORAGE_CONNECTION_STRING",
                "DefaultEndpointsProtocol=https;AccountName=devaccount;AccountKey=devkey;EndpointSuffix=core.windows.net"
            ),
            event_hub_name=os.getenv("EVENT_HUB_NAME", "fraud-transactions"),
            event_hub_connection_string=os.getenv(
                "EVENT_HUB_CONNECTION_STRING",
                "Endpoint=sb://dev.servicebus.windows.net/;SharedAccessKeyName=RootManageSharedAccessKey;SharedAccessKey=devkey"
            ),
            storage_account_name=os.getenv("AZURE_STORAGE_ACCOUNT", "frauddetectiondl"),
            storage_account_key=os.getenv("AZURE_STORAGE_KEY", "devkey"),
            data_lake_path=os.getenv("AZURE_DATA_LAKE_PATH", "/data/transactions")
        )

        # Database
        self.database = DatabaseSettings(
            driver=os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server"),
            server=os.getenv("DB_SERVER", "localhost"),
            database=os.getenv("DB_NAME", "frauddb"),
            username=os.getenv("DB_USER", "sa"),
            password=os.getenv("DB_PASSWORD", "YourPassword123!"),
            port=int(os.getenv("DB_PORT", "1433")),
            pool_size=int(os.getenv("DB_POOL_SIZE", "20")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "40"))
        )

        # Kafka (optional)
        self.kafka = KafkaSettings(
            bootstrap_servers=os.getenv("KAFKA_BROKERS", "localhost:9092").split(","),
            topic=os.getenv("KAFKA_TOPIC", "fraud-transactions"),
            group_id=os.getenv("KAFKA_GROUP_ID", "fraud-detection-group"),
            consumer_timeout_ms=int(os.getenv("KAFKA_TIMEOUT_MS", "3000"))
        )

        # Cache
        self.cache = CacheSettings(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD"),
            ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "3600"))
        )

        # Monitoring
        self.monitoring = MonitoringSettings(
            prometheus_port=int(os.getenv("PROMETHEUS_PORT", "8000")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            enable_profiling=os.getenv("ENABLE_PROFILING", "false").lower() == "true",
            enable_data_validation=os.getenv("ENABLE_DATA_VALIDATION", "true").lower() == "true"
        )

    @property
    def database_url(self) -> str:
        """Construct database connection URL for SQLAlchemy"""
        return (
            f"mssql+pyodbc://{self.database.username}:{self.database.password}"
            f"@{self.database.server}:{self.database.port}/{self.database.database}"
            f"?driver={self.database.driver.replace(' ', '+')}"
        )

    def __repr__(self) -> str:
        return (
            f"Settings(env={self.env}, debug={self.debug}, "
            f"database={self.database.server}, cache={self.cache.host})"
        )


# Singleton instance
settings = Settings()

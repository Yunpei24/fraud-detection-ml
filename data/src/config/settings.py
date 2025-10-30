"""
Configuration settings for the data pipeline
Migrated to use centralized configuration for consistency.
"""

import sys
from pathlib import Path
from typing import Optional

# Add project root to path to import centralized config
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from config import get_settings

# Get centralized settings
settings = get_settings()


# Azure settings
class AzureSettings:
    """Azure cloud configuration"""

    connection_string: str = settings.azure.storage_connection_string
    event_hub_name: str = settings.azure.event_hub_name
    event_hub_connection_string: Optional[
        str
    ] = settings.azure.event_hub_connection_string
    storage_account: str = settings.azure.storage_account_name
    storage_key: str = settings.azure.storage_account_key
    data_lake_path: str = settings.azure.data_lake_path


# Database settings
class DatabaseSettings:
    """Database configuration"""

    server: str = settings.database.host
    database: str = settings.database.name
    username: str = settings.database.user
    password: str = settings.database.password
    port: int = settings.database.port
    pool_size: int = settings.database.pool_size
    max_overflow: int = settings.database.max_overflow


# Kafka settings
class KafkaSettings:
    """Kafka configuration (alternative to Event Hub)"""

    bootstrap_servers: str = settings.kafka.bootstrap_servers
    topic: str = settings.kafka.topic_transactions
    group_id: str = settings.kafka.consumer_group
    consumer_timeout_ms: int = settings.kafka.timeout_ms

    @property
    def bootstrap_servers_list(self) -> list:
        """Convert comma-separated servers to list"""
        return settings.kafka.bootstrap_servers_list


# Cache settings
class CacheSettings:
    """Redis cache configuration"""

    host: str = settings.redis.host
    port: int = settings.redis.port
    db: int = settings.redis.db
    password: Optional[str] = settings.redis.password
    ttl_seconds: int = settings.redis.ttl_seconds


# Monitoring settings
class MonitoringSettings:
    """Monitoring and observability"""

    prometheus_port: int = settings.data.prometheus_port
    log_level: str = settings.monitoring.log_level
    enable_profiling: bool = settings.data.enable_profiling
    enable_data_validation: bool = settings.data.enable_data_validation


# Main settings class
class Settings:
    """
    Main settings class that loads configuration from centralized config
    """

    # Environment
    env: str = settings.environment
    debug: bool = settings.debug

    # Nested settings (instantiated on access)
    azure: AzureSettings = AzureSettings()
    database: DatabaseSettings = DatabaseSettings()
    kafka: KafkaSettings = KafkaSettings()
    cache: CacheSettings = CacheSettings()
    monitoring: MonitoringSettings = MonitoringSettings()

    @property
    def database_url(self) -> str:
        """Construct database connection URL for SQLAlchemy"""
        return settings.database.url

    def __repr__(self) -> str:
        return (
            f"Settings(env={self.env}, debug={self.debug}, "
            f"database={self.database.server}, cache={self.cache.host})"
        )


# Singleton instance
settings_instance = Settings()

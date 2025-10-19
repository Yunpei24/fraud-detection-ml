"""
Feature Store service - stores computed features for model training
Integrates with Tecton, Feast, or custom feature store
"""

import logging
from typing import Optional, Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class FeatureStoreService:
    """
    Service for managing features for ML models
    Handles online and offline features
    """

    def __init__(
        self,
        store_type: str = "online",  # "online" or "offline"
        backend: str = "redis"  # "redis", "postgres", or other
    ):
        """
        Initialize Feature Store service
        
        Args:
            store_type: Type of feature store (online/offline)
            backend: Backend storage system
        """
        self.store_type = store_type
        self.backend = backend
        self.client = None
        self._initialized = False

    def connect(self) -> None:
        """Establish connection to feature store"""
        try:
            if self.backend == "redis":
                import redis
                from ..config.settings import settings
                
                self.client = redis.Redis(
                    host=settings.cache.host,
                    port=settings.cache.port,
                    db=settings.cache.db,
                    password=settings.cache.password,
                    decode_responses=True
                )
                self.client.ping()
                self._initialized = True
                logger.info("Connected to Redis feature store")
            else:
                logger.warning(f"Backend '{self.backend}' not yet implemented")

        except Exception as e:
            logger.error(f"Failed to connect to feature store: {str(e)}")
            raise

    def disconnect(self) -> None:
        """Close feature store connection"""
        if self.client:
            if self.backend == "redis":
                self.client.close()
            self._initialized = False
            logger.info("Disconnected from feature store")

    def save_features(
        self,
        entity_id: str,
        features: Dict[str, Any],
        ttl_seconds: int = 3600
    ) -> bool:
        """
        Save features for an entity (customer/merchant/transaction)
        
        Args:
            entity_id: Entity identifier (customer_id, merchant_id, etc)
            features: Dictionary of feature names and values
            ttl_seconds: Time-to-live for features
        
        Returns:
            Success status
        """
        if not self._initialized:
            self.connect()

        try:
            if self.backend == "redis":
                import json
                
                # Create feature key
                feature_key = f"features:{entity_id}"
                
                # Store as JSON
                self.client.setex(
                    feature_key,
                    ttl_seconds,
                    json.dumps(features)
                )
                
                logger.debug(f"Saved features for entity: {entity_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to save features: {str(e)}")
            return False

    def get_features(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve features for an entity
        
        Args:
            entity_id: Entity identifier
        
        Returns:
            Dictionary of features or None if not found
        """
        if not self._initialized:
            self.connect()

        try:
            if self.backend == "redis":
                import json
                
                feature_key = f"features:{entity_id}"
                features_json = self.client.get(feature_key)
                
                if features_json:
                    features = json.loads(features_json)
                    logger.debug(f"Retrieved features for entity: {entity_id}")
                    return features
                
                logger.debug(f"No features found for entity: {entity_id}")
                return None

        except Exception as e:
            logger.error(f"Failed to get features: {str(e)}")
            return None

    def batch_save_features(
        self,
        features_dict: Dict[str, Dict[str, Any]],
        ttl_seconds: int = 3600
    ) -> int:
        """
        Save multiple entities' features efficiently
        
        Args:
            features_dict: {entity_id: {feature_name: value}}
            ttl_seconds: Time-to-live for features
        
        Returns:
            Number of entities saved
        """
        if not self._initialized:
            self.connect()

        saved_count = 0

        try:
            for entity_id, features in features_dict.items():
                if self.save_features(entity_id, features, ttl_seconds):
                    saved_count += 1

            logger.info(f"Batch saved features for {saved_count} entities")
            return saved_count

        except Exception as e:
            logger.error(f"Failed in batch save: {str(e)}")
            return saved_count

    def delete_features(self, entity_id: str) -> bool:
        """
        Delete features for an entity
        
        Args:
            entity_id: Entity identifier
        
        Returns:
            Success status
        """
        if not self._initialized:
            self.connect()

        try:
            if self.backend == "redis":
                feature_key = f"features:{entity_id}"
                self.client.delete(feature_key)
                logger.debug(f"Deleted features for entity: {entity_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to delete features: {str(e)}")
            return False

    def exists(self, entity_id: str) -> bool:
        """
        Check if features exist for an entity
        
        Args:
            entity_id: Entity identifier
        
        Returns:
            True if features exist
        """
        if not self._initialized:
            self.connect()

        try:
            if self.backend == "redis":
                feature_key = f"features:{entity_id}"
                return self.client.exists(feature_key) > 0

        except Exception as e:
            logger.error(f"Failed to check feature existence: {str(e)}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get feature store statistics
        
        Returns:
            Statistics dictionary
        """
        if not self._initialized:
            self.connect()

        try:
            if self.backend == "redis":
                info = self.client.info()
                return {
                    "used_memory": info.get('used_memory_human'),
                    "connected_clients": info.get('connected_clients'),
                    "total_commands_processed": info.get('total_commands_processed')
                }

        except Exception as e:
            logger.error(f"Failed to get statistics: {str(e)}")
            return {}

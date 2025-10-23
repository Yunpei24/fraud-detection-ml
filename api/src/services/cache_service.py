"""
Cache service using Redis.
"""
import json
from typing import Optional, Dict, Any
import redis

from ..config import get_logger, settings, constants
from ..utils import CacheException

logger = get_logger(__name__)


class CacheService:
    """Service for caching predictions using Redis."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """
        Initialize cache service.
        
        Args:
            redis_client: Redis client instance (optional)
        """
        self.redis_client = redis_client
        self.logger = logger
        self.ttl = constants.CACHE_TTL_SECONDS
    
    def _get_client(self) -> redis.Redis:
        """
        Get or create Redis client.
        
        Returns:
            Redis client
            
        Raises:
            CacheException: If Redis connection fails
        """
        if self.redis_client is None:
            try:
                self.redis_client = redis.from_url(
                    settings.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5
                )
                # Test connection
                self.redis_client.ping()
            except Exception as e:
                self.logger.error(f"Failed to connect to Redis: {e}")
                raise CacheException(
                    "Failed to connect to Redis",
                    details={"error": str(e)}
                )
        
        return self.redis_client
    
    async def get_cached_prediction(
        self,
        transaction_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached prediction for a transaction.
        
        Args:
            transaction_id: Transaction identifier
            
        Returns:
            Cached prediction or None if not found
        """
        if not settings.enable_cache:
            return None
        
        try:
            client = self._get_client()
            cache_key = self._make_key(transaction_id)
            
            cached_data = client.get(cache_key)
            
            if cached_data:
                self.logger.info(f"Cache hit for transaction {transaction_id}")
                return json.loads(cached_data)
            
            self.logger.debug(f"Cache miss for transaction {transaction_id}")
            return None
            
        except redis.RedisError as e:
            self.logger.warning(f"Redis error when getting cache: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in cache get: {e}")
            return None
    
    async def set_prediction_cache(
        self,
        transaction_id: str,
        prediction: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache a prediction result.
        
        Args:
            transaction_id: Transaction identifier
            prediction: Prediction data to cache
            ttl: Time to live in seconds (optional)
            
        Returns:
            True if cached successfully
        """
        if not settings.enable_cache:
            return False
        
        try:
            client = self._get_client()
            cache_key = self._make_key(transaction_id)
            
            # Serialize prediction
            prediction_json = json.dumps(prediction)
            
            # Set with TTL
            ttl_seconds = ttl or self.ttl
            client.setex(cache_key, ttl_seconds, prediction_json)
            
            self.logger.debug(
                f"Cached prediction for transaction {transaction_id} "
                f"(TTL: {ttl_seconds}s)"
            )
            return True
            
        except redis.RedisError as e:
            self.logger.warning(f"Redis error when setting cache: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error in cache set: {e}")
            return False
    
    async def invalidate_cache(self, transaction_id: str) -> bool:
        """
        Invalidate cached prediction.
        
        Args:
            transaction_id: Transaction identifier
            
        Returns:
            True if invalidated successfully
        """
        try:
            client = self._get_client()
            cache_key = self._make_key(transaction_id)
            
            deleted = client.delete(cache_key)
            
            if deleted:
                self.logger.info(f"Invalidated cache for transaction {transaction_id}")
            
            return bool(deleted)
            
        except redis.RedisError as e:
            self.logger.warning(f"Redis error when invalidating cache: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error in cache invalidation: {e}")
            return False
    
    async def clear_all_cache(self) -> bool:
        """
        Clear all cached predictions.
        
        Returns:
            True if cleared successfully
        """
        try:
            client = self._get_client()
            
            # Get all keys with our prefix
            pattern = self._make_key("*")
            keys = client.keys(pattern)
            
            if keys:
                deleted = client.delete(*keys)
                self.logger.info(f"Cleared {deleted} cached predictions")
            
            return True
            
        except redis.RedisError as e:
            self.logger.warning(f"Redis error when clearing cache: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error in cache clear: {e}")
            return False
    
    def check_health(self) -> bool:
        """
        Check Redis connection health.
        
        Returns:
            True if Redis is healthy
        """
        try:
            client = self._get_client()
            client.ping()
            return True
        except Exception as e:
            self.logger.error(f"Redis health check failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        try:
            client = self._get_client()
            info = client.info("stats")
            
            return {
                "total_connections_received": info.get("total_connections_received", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "instantaneous_ops_per_sec": info.get("instantaneous_ops_per_sec", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0)
            }
        except Exception as e:
            self.logger.error(f"Failed to get Redis stats: {e}")
            return {}
    
    def _make_key(self, transaction_id: str) -> str:
        """
        Generate Redis cache key.
        
        Args:
            transaction_id: Transaction identifier
            
        Returns:
            Redis key string
        """
        return f"fraud_detection:prediction:{transaction_id}"

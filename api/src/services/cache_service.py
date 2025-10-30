"""
Cache service using Redis.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import redis

from ..config import constants, get_logger, settings
from ..utils import CacheException

logger = get_logger(__name__)


# Exception classes
class CacheConnectionError(CacheException):
    """Exception raised when cache connection fails."""

    pass


class CacheOperationError(CacheException):
    """Exception raised when cache operation fails."""

    pass


# Model classes
@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    value: Any
    timestamp: Optional[datetime] = None
    ttl: Optional[int] = None


class CacheService:
    """Service for caching predictions using Redis."""

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """
        Initialize cache service.

        Args:
            redis_client: Optional Redis client instance
        """
        self.redis_client = redis_client
        self.default_ttl = 3600  # Default 1 hour TTL
        self.logger = logger

    def connect(self) -> redis.Redis:
        """
        Get Redis client.

        Returns:
            Redis client

        Raises:
            CacheConnectionError: If connection fails
        """
        if self.redis_client is None:
            raise CacheConnectionError("No Redis client provided")

        return self.redis_client

    async def get_cached_prediction(
        self, transaction_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached prediction for a transaction.

        Args:
            transaction_id: Transaction identifier

        Returns:
            Cached prediction or None if not found
        """
        if self.redis_client is None:
            return None

        try:
            client = self.connect()
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
        self, transaction_id: str, prediction: Dict[str, Any], ttl: Optional[int] = None
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
        if self.redis_client is None:
            return False

        try:
            client = self.connect()
            cache_key = self._make_key(transaction_id)

            # Serialize prediction
            prediction_json = json.dumps(prediction)

            # Set with TTL
            ttl_seconds = ttl or self.default_ttl
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
            client = self.connect()
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
            client = self.connect()

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
            client = self.connect()
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
            client = self.connect()
            info = client.info("stats")

            return {
                "total_connections_received": info.get("total_connections_received", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "instantaneous_ops_per_sec": info.get("instantaneous_ops_per_sec", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
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

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """
        Set a string value.

        Args:
            key: Cache key
            value: String value
            ttl: Time to live in seconds

        Returns:
            True if successful
        """
        try:
            client = self.connect()
            if ttl is not None:
                client.set(key, value, ex=ttl)
            else:
                client.set(key, value, ex=None)
            self.logger.debug(f"Set cache key: {key}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to set cache: {e}")
            raise CacheOperationError(f"Failed to set cache: {str(e)}")

    def get(self, key: str) -> Optional[str]:
        """
        Get a string value.

        Args:
            key: Cache key

        Returns:
            String value or None
        """
        try:
            client = self.connect()
            value = client.get(key)
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return value
        except Exception as e:
            self.logger.error(f"Failed to get cache: {e}")
            return None

    def set_json(
        self, key: str, value: Dict[str, Any], ttl: Optional[int] = None
    ) -> bool:
        """
        Set a JSON value.

        Args:
            key: Cache key
            value: Dictionary value
            ttl: Time to live in seconds

        Returns:
            True if successful
        """
        try:
            json_str = json.dumps(value)
            return self.set(key, json_str, ttl)
        except Exception as e:
            self.logger.error(f"Failed to set JSON cache: {e}")
            raise CacheOperationError(f"Failed to set JSON cache: {str(e)}")

    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get a JSON value.

        Args:
            key: Cache key

        Returns:
            Dictionary value or None
        """
        try:
            json_str = self.get(key)
            if json_str:
                return json.loads(json_str)
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON cache: {e}")
            raise CacheOperationError(f"Invalid JSON in cache: {str(e)}")
        except Exception as e:
            self.logger.error(f"Failed to get JSON cache: {e}")
            return None

    def delete(self, key: str) -> bool:
        """
        Delete a key.

        Args:
            key: Cache key

        Returns:
            True if key was deleted
        """
        try:
            client = self.connect()
            result = client.delete(key)
            return bool(result)
        except Exception as e:
            self.logger.error(f"Failed to delete cache: {e}")
            return False

    def exists(self, key: str) -> bool:
        """
        Check if key exists.

        Args:
            key: Cache key

        Returns:
            True if key exists
        """
        try:
            client = self.connect()
            return bool(client.exists(key))
        except Exception as e:
            self.logger.error(f"Failed to check cache existence: {e}")
            return False

    def expire(self, key: str, ttl: int) -> bool:
        """
        Set expiration on key.

        Args:
            key: Cache key
            ttl: Time to live in seconds

        Returns:
            True if successful
        """
        try:
            client = self.connect()
            return bool(client.expire(key, ttl))
        except Exception as e:
            self.logger.error(f"Failed to set cache expiration: {e}")
            return False

    def get_ttl(self, key: str) -> int:
        """
        Get TTL for key.

        Args:
            key: Cache key

        Returns:
            TTL in seconds (-2 if key doesn't exist, -1 if no expiration)
        """
        try:
            client = self.connect()
            return client.ttl(key)
        except Exception as e:
            self.logger.error(f"Failed to get cache TTL: {e}")
            return -2

    def increment(self, key: str, amount: int = 1) -> int:
        """
        Increment a counter.

        Args:
            key: Counter key
            amount: amount to increment

        Returns:
            New counter value
        """
        try:
            client = self.connect()
            if amount == 1:
                return client.incr(key)
            else:
                return client.incrby(key, amount)
        except Exception as e:
            self.logger.error(f"Failed to increment cache: {e}")
            raise CacheOperationError(f"Failed to increment cache: {str(e)}")

    def hset(self, key: str, mapping: Dict[str, str]) -> bool:
        """
        Set hash fields.

        Args:
            key: Hash key
            mapping: Field-value mapping

        Returns:
            True if successful
        """
        try:
            client = self.connect()
            client.hset(key, mapping=mapping)
            return True
        except Exception as e:
            self.logger.error(f"Failed to set hash cache: {e}")
            raise CacheOperationError(f"Failed to set hash cache: {str(e)}")

    def hgetall(self, key: str) -> Dict[str, str]:
        """
        Get all hash fields.

        Args:
            key: Hash key

        Returns:
            Field-value mapping
        """
        try:
            client = self.connect()
            result = client.hgetall(key)
            return {
                k.decode("utf-8")
                if isinstance(k, bytes)
                else k: (v.decode("utf-8") if isinstance(v, bytes) else v)
                for k, v in result.items()
            }
        except Exception as e:
            self.logger.error(f"Failed to get hash cache: {e}")
            return {}

    def lpush(self, key: str, *values: str) -> bool:
        """
        Push values to list.

        Args:
            key: List key
            values: Values to push

        Returns:
            True if successful
        """
        try:
            client = self.connect()
            client.lpush(key, *values)
            return True
        except Exception as e:
            self.logger.error(f"Failed to push to list cache: {e}")
            raise CacheOperationError(f"Failed to push to list cache: {str(e)}")

    def rpop(self, key: str) -> Optional[str]:
        """
        Pop value from list.

        Args:
            key: List key

        Returns:
            Popped value or None
        """
        try:
            client = self.connect()
            result = client.rpop(key)
            if isinstance(result, bytes):
                return result.decode("utf-8")
            return result
        except Exception as e:
            self.logger.error(f"Failed to pop from list cache: {e}")
            return None

    def sadd(self, key: str, *members: str) -> bool:
        """
        Add members to set.

        Args:
            key: Set key
            members: Members to add

        Returns:
            True if successful
        """
        try:
            client = self.connect()
            client.sadd(key, *members)
            return True
        except Exception as e:
            self.logger.error(f"Failed to add to set cache: {e}")
            raise CacheOperationError(f"Failed to add to set cache: {str(e)}")

    def sismember(self, key: str, member: str) -> bool:
        """
        Check if member is in set.

        Args:
            key: Set key
            member: Member to check

        Returns:
            True if member exists
        """
        try:
            client = self.connect()
            return bool(client.sismember(key, member))
        except Exception as e:
            self.logger.error(f"Failed to check set membership: {e}")
            return False

    def clear_pattern(self, pattern: str) -> int:
        """
        Clear keys matching pattern.

        Args:
            pattern: Key pattern

        Returns:
            Number of keys deleted
        """
        try:
            client = self.connect()
            keys = list(client.scan_iter(pattern))
            if keys:
                return client.delete(*keys)
            return 0
        except Exception as e:
            self.logger.error(f"Failed to clear pattern: {e}")
            return 0

    def cache_prediction_result(self, cache_entry: CacheEntry) -> bool:
        """
        Cache a prediction result.

        Args:
            cache_entry: Cache entry with prediction data

        Returns:
            True if successful
        """
        return self.set_json(cache_entry.key, cache_entry.value, cache_entry.ttl)

    def get_cached_prediction(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached prediction.

        Args:
            key: Cache key

        Returns:
            Prediction data or None
        """
        return self.get_json(key)

    def mset(self, mapping: Dict[str, str]) -> bool:
        """
        Set multiple keys.

        Args:
            mapping: Key-value mapping

        Returns:
            True if successful
        """
        try:
            client = self.connect()
            client.mset(mapping)
            return True
        except Exception as e:
            self.logger.error(f"Failed to set multiple cache: {e}")
            raise CacheOperationError(f"Failed to set multiple cache: {str(e)}")

    def mget(self, keys: List[str]) -> List[Optional[str]]:
        """
        Get multiple keys.

        Args:
            keys: List of keys

        Returns:
            List of values
        """
        try:
            client = self.connect()
            results = client.mget(keys)
            return [r.decode("utf-8") if isinstance(r, bytes) else r for r in results]
        except Exception as e:
            self.logger.error(f"Failed to get multiple cache: {e}")
            return [None] * len(keys)

    def pipeline(self):
        """
        Get Redis pipeline.

        Returns:
            Redis pipeline context manager
        """
        client = self.connect()
        return client.pipeline()

    def health_check(self) -> Dict[str, Any]:
        """
        Check cache health.

        Returns:
            Health status dictionary
        """
        try:
            client = self.connect()
            client.ping()
            return {"status": "healthy", "connection": "ok"}
        except Exception as e:
            return {"status": "unhealthy", "connection_error": str(e)}

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Statistics dictionary
        """
        try:
            client = self.connect()
            info = client.info("stats")
            hits = info.get("keyspace_hits", 0)
            misses = info.get("keyspace_misses", 0)
            total = hits + misses
            hit_rate = hits / total if total > 0 else 0

            return {
                "hits": hits,
                "misses": misses,
                "hit_rate": hit_rate,
                "total_requests": total,
            }
        except Exception as e:
            self.logger.error(f"Failed to get cache statistics: {e}")
            return {}

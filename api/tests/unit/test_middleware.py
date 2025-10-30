from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.api.middleware import (
    ErrorHandlingMiddleware,
    RateLimitingMiddleware,
    RequestLoggingMiddleware,
)


class TestRequestLoggingMiddleware:
    @pytest.mark.asyncio
    async def test_middleware_logs_request(self):
        middleware = RequestLoggingMiddleware(MagicMock())

        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.url.path = "/api/test"
        mock_request.client.host = "127.0.0.1"
        mock_request.headers.get.return_value = "test-id-123"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}

        async def call_next(req):
            return mock_response

        with patch("logging.Logger.info") as mock_logger:
            result = await middleware.dispatch(mock_request, call_next)

            assert result is mock_response
            assert "X-Request-ID" in result.headers
            assert "X-Process-Time" in result.headers

    @pytest.mark.asyncio
    async def test_middleware_handles_errors(self):
        middleware = RequestLoggingMiddleware(MagicMock())

        mock_request = MagicMock()
        mock_request.method = "GET"
        mock_request.url.path = "/api/test"
        mock_request.client.host = "127.0.0.1"
        mock_request.headers.get.return_value = "test-id"

        async def call_next_error(req):
            raise Exception("Test error")

        with pytest.raises(Exception):
            await middleware.dispatch(mock_request, call_next_error)


class TestErrorHandlingMiddleware:
    @pytest.mark.asyncio
    async def test_middleware_catches_validation_error(self):
        middleware = ErrorHandlingMiddleware(MagicMock())

        mock_request = MagicMock()
        mock_request.url.path = "/api/test"

        async def call_next(req):
            raise ValueError("Invalid input")

        response = await middleware.dispatch(mock_request, call_next)

        assert response.status_code == 400
        assert b"VALIDATION_ERROR" in response.body

    @pytest.mark.asyncio
    async def test_middleware_catches_permission_error(self):
        middleware = ErrorHandlingMiddleware(MagicMock())

        mock_request = MagicMock()
        mock_request.url.path = "/api/test"

        async def call_next(req):
            raise PermissionError("Not allowed")

        response = await middleware.dispatch(mock_request, call_next)

        assert response.status_code == 403
        assert b"PERMISSION_DENIED" in response.body

    @pytest.mark.asyncio
    async def test_middleware_catches_generic_error(self):
        middleware = ErrorHandlingMiddleware(MagicMock())

        mock_request = MagicMock()
        mock_request.url.path = "/api/test"

        async def call_next(req):
            raise RuntimeError("Server error")

        response = await middleware.dispatch(mock_request, call_next)

        assert response.status_code == 500
        assert b"INTERNAL_SERVER_ERROR" in response.body


class TestRateLimitingMiddleware:
    def test_init_without_redis(self):
        middleware = RateLimitingMiddleware(MagicMock(), redis_client=None)
        assert middleware.redis_client is None
        assert middleware.requests_per_minute == 1000

    def test_init_with_redis(self):
        mock_redis = MagicMock()
        middleware = RateLimitingMiddleware(
            MagicMock(), redis_client=mock_redis, requests_per_minute=100
        )
        assert middleware.redis_client is mock_redis
        assert middleware.requests_per_minute == 100

    @pytest.mark.asyncio
    async def test_rate_limiting_pass_through_without_redis(self):
        middleware = RateLimitingMiddleware(MagicMock(), redis_client=None)

        mock_request = MagicMock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers.get.return_value = None

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}

        async def call_next(req):
            return mock_response

        result = await middleware.dispatch(mock_request, call_next)
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_rate_limiting_with_redis_under_limit(self):
        mock_redis = MagicMock()
        mock_redis.incr.return_value = 50

        middleware = RateLimitingMiddleware(
            MagicMock(), redis_client=mock_redis, requests_per_minute=100
        )

        mock_request = MagicMock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers.get.return_value = None

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}

        async def call_next(req):
            return mock_response

        result = await middleware.dispatch(mock_request, call_next)
        assert result.status_code == 200
        assert "X-RateLimit-Limit" in result.headers

    @pytest.mark.asyncio
    async def test_rate_limiting_exceeds_limit(self):
        mock_redis = MagicMock()
        mock_redis.incr.return_value = 1001

        middleware = RateLimitingMiddleware(
            MagicMock(), redis_client=mock_redis, requests_per_minute=1000
        )

        mock_request = MagicMock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers.get.return_value = None

        async def call_next(req):
            pass

        response = await middleware.dispatch(mock_request, call_next)
        assert response.status_code == 429
        assert b"RATE_LIMIT_EXCEEDED" in response.body

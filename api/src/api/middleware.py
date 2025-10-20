import logging
import time
from datetime import datetime
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        request_id = request.headers.get("x-request-id", f"{int(start_time * 1000000)}")
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path

        logger.info(
            f"REQUEST | id={request_id} | {method} {path} | client={client_ip}",
            extra={
                "request_id": request_id,
                "method": method,
                "path": path,
                "client_ip": client_ip,
            },
        )

        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            logger.info(
                f"RESPONSE | id={request_id} | status={response.status_code} | "
                f"time={process_time:.3f}s",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "process_time": process_time,
                },
            )

            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)

            return response

        except Exception as e:
            logger.error(
                f"REQUEST ERROR | id={request_id} | {method} {path} | error={str(e)}",
                exc_info=True,
                extra={"request_id": request_id, "method": method, "path": path},
            )
            raise


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response

        except ValueError as e:
            logger.warning(f"Validation error: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={
                    "error_code": "VALIDATION_ERROR",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

        except PermissionError as e:
            logger.warning(f"Permission error: {str(e)}")
            return JSONResponse(
                status_code=403,
                content={
                    "error_code": "PERMISSION_DENIED",
                    "message": "Access denied",
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

        except Exception as e:
            logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error_code": "INTERNAL_SERVER_ERROR",
                    "message": "An internal error occurred",
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )


class RateLimitingMiddleware(BaseHTTPMiddleware):
    def __init__(
        self, app, redis_client=None, requests_per_minute: int = 1000
    ):
        super().__init__(app)
        self.redis_client = redis_client
        self.requests_per_minute = requests_per_minute
        self.window = 60

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        api_key = request.headers.get("x-api-key", None)
        if api_key:
            identifier = f"api_key:{api_key}"
        else:
            identifier = f"ip:{request.client.host}" if request.client else "unknown"

        key = f"rate_limit:{identifier}"
        current_count = 0

        if self.redis_client:
            try:
                current_count = self.redis_client.incr(key)

                if current_count == 1:
                    self.redis_client.expire(key, self.window)

                if current_count > self.requests_per_minute:
                    logger.warning(
                        f"Rate limit exceeded for {identifier}: "
                        f"{current_count}/{self.requests_per_minute}"
                    )
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error_code": "RATE_LIMIT_EXCEEDED",
                            "message": f"Rate limit exceeded ({self.requests_per_minute} requests/min)",
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    )

            except Exception as e:
                logger.error(f"Redis error in rate limiting: {str(e)}")

        response = await call_next(request)

        if self.redis_client and current_count > 0:
            response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
            response.headers["X-RateLimit-Remaining"] = str(
                max(0, self.requests_per_minute - current_count)
            )
            response.headers["X-RateLimit-Reset"] = str(int(time.time()) + self.window)

        return response


class CORSMiddleware(BaseHTTPMiddleware):
    def __init__(
        self, app, allow_origins: list = None, allow_methods: list = None, allow_headers: list = None
    ):
        super().__init__(app)
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allow_headers = allow_headers or ["*"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.method == "OPTIONS":
            return Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": ", ".join(self.allow_origins),
                    "Access-Control-Allow-Methods": ", ".join(self.allow_methods),
                    "Access-Control-Allow-Headers": ", ".join(self.allow_headers),
                    "Access-Control-Max-Age": "3600",
                },
            )

        response = await call_next(request)

        response.headers["Access-Control-Allow-Origin"] = ", ".join(self.allow_origins)
        response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
        response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allow_headers)

        return response

import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.routes import (
    admin_router,
    audit_router,
    auth_router,
    deployment_router,
    drift_router,
    explain_router,
    health_router,
    metrics_router,
    predict_router,
    transaction_router,
    users_router,
)
from .config import constants, get_logger, settings
from .monitoring.prometheus import (
    ACTIVE_CONNECTIONS,
    API_ERRORS_TOTAL,
    API_REQUEST_DURATION_SECONDS,
    API_REQUESTS_TOTAL,
    INVALID_REQUESTS_TOTAL,
)
from .utils import FraudDetectionException

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        f"Starting Fraud Detection API v{constants.API_VERSION}",
        extra={"environment": settings.environment, "log_level": settings.log_level},
    )
    yield
    logger.info("Shutting down Fraud Detection API")


app = FastAPI(
    title="Fraud Detection API",
    description="ML-powered fraud detection system with ensemble models",
    version=constants.API_VERSION,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=settings.api.cors_allow_credentials,
    allow_methods=settings.api.cors_allow_methods,
    allow_headers=settings.api.cors_allow_headers,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    ACTIVE_CONNECTIONS.inc()

    response = await call_next(request)

    process_time = time.time() - start_time

    logger.info(
        f"{request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "process_time": process_time,
            "client_ip": request.client.host if request.client else None,
        },
    )

    response.headers["X-Process-Time"] = str(process_time)

    # Update Prometheus metrics with correct labels
    API_REQUESTS_TOTAL.labels(
        endpoint=request.url.path,
        method=request.method,
        status_code=response.status_code,
    ).inc()

    API_REQUEST_DURATION_SECONDS.labels(
        endpoint=request.url.path, method=request.method
    ).observe(process_time)

    ACTIVE_CONNECTIONS.dec()

    return response


@app.exception_handler(FraudDetectionException)
async def fraud_detection_exception_handler(
    request: Request, exc: FraudDetectionException
):
    API_ERRORS_TOTAL.labels(
        error_type=exc.__class__.__name__, error_code=exc.error_code
    ).inc()

    logger.error(
        f"FraudDetectionException: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "details": exc.details,
            "path": request.url.path,
        },
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_code": exc.error_code,
            "message": exc.message,
            "details": exc.details,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    INVALID_REQUESTS_TOTAL.labels(validation_error=exc.__class__.__name__).inc()

    logger.warning(
        f"Validation error: {exc}",
        extra={"path": request.url.path, "errors": exc.errors()},
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error_code": "E001",
            "message": "Validation error",
            "details": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    API_ERRORS_TOTAL.labels(error_type=exc.__class__.__name__, error_code="E999").inc()

    logger.error(
        f"Unhandled exception: {exc}",
        exc_info=True,
        extra={"path": request.url.path},
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_code": "E999",
            "message": "Internal server error",
            "details": (
                {"error": str(exc)} if settings.environment != "production" else {}
            ),
        },
    )


app.include_router(health_router)
app.include_router(auth_router)  # Authentication endpoints
app.include_router(users_router)  # User management endpoints (admin only)
app.include_router(deployment_router)  # Deployment management endpoints (admin only)
app.include_router(predict_router)
app.include_router(metrics_router)
app.include_router(admin_router)
app.include_router(explain_router)
app.include_router(drift_router)
app.include_router(audit_router)
app.include_router(transaction_router)


@app.get("/", tags=["root"])
async def root():
    return {
        "name": "Fraud Detection API",
        "version": constants.API_VERSION,
        "status": "running",
        "environment": settings.environment,
        "docs": "/docs" if settings.environment != "production" else "disabled",
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.environment == "development",
        log_level=settings.monitoring.log_level.lower(),
    )

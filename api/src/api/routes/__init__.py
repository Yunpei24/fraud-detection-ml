"""
API routes package.
"""
from .predict import router as predict_router
from .health import router as health_router
from .metrics import router as metrics_router
from .admin import router as admin_router

__all__ = [
    "predict_router",
    "health_router",
    "metrics_router",
    "admin_router"
]

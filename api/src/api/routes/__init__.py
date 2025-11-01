"""
API routes package.
"""

from .admin import router as admin_router
from .audit import router as audit_router
from .auth import router as auth_router
from .drift import router as drift_router
from .explain import router as explain_router
from .health import router as health_router
from .metrics import router as metrics_router
from .predict import router as predict_router
from .transaction import router as transaction_router
from .users import router as users_router

__all__ = [
    "predict_router",
    "health_router",
    "metrics_router",
    "admin_router",
    "explain_router",
    "drift_router",
    "audit_router",
    "transaction_router",
    "auth_router",
    "users_router",
]

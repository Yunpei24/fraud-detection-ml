# training/src/cloud/__init__.py
from .databricks import get_spark_session, submit_job
from .azure import upload_to_blob, download_from_blob, get_secret_from_vault

__all__ = [
    "get_spark_session",
    "submit_job",
    "upload_to_blob",
    "download_from_blob",
    "get_secret_from_vault",
]

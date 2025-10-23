# training/src/cloud/databricks.py
from __future__ import annotations

import os
from typing import Optional, Dict, Any

try:
    from pyspark.sql import SparkSession  # type: ignore
    _HAS_SPARK = True
except Exception:
    SparkSession = None  # type: ignore
    _HAS_SPARK = False


def get_spark_session(app_name: str = "fraud_detection_training") -> Optional[Any]:
    """
    Create or retrieve a SparkSession.
    Works both locally and on Databricks clusters.
    """
    if not _HAS_SPARK:
        print("[cloud.databricks] Spark not available - returning None")
        return None

    builder = SparkSession.builder.appName(app_name)
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        # Running on a Databricks cluster
        spark = builder.getOrCreate()
    else:
        # Local fallback
        builder = builder.master("local[*]")
        spark = builder.getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    print(f"[cloud.databricks] SparkSession initialized ({spark.version})")
    return spark


def submit_job(job_name: str, notebook_path: str, parameters: Optional[Dict[str, Any]] = None) -> None:
    """
    Simulated Databricks job submission.
    In production, you can integrate with Databricks REST API (2.1) using requests.
    """
    print(f"[cloud.databricks] Submitting job: {job_name}")
    print(f"  Notebook path: {notebook_path}")
    if parameters:
        print(f"  Parameters: {parameters}")
    else:
        print("  No parameters provided.")

    # Example (uncomment in real use):
    # import requests
    # url = "https://<databricks-instance>/api/2.1/jobs/run-now"
    # headers = {"Authorization": f"Bearer {os.environ['DATABRICKS_TOKEN']}"}
    # payload = {"job_id": <job_id>, "notebook_params": parameters}
    # requests.post(url, headers=headers, json=payload)

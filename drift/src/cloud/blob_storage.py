"""
Azure Blob Storage utilities for drift detection reports.
"""

from azure.storage.blob import BlobServiceClient
from typing import List, Optional
from pathlib import Path
import structlog

from ..config.settings import Settings

logger = structlog.get_logger(__name__)


def get_blob_service_client(settings: Optional[Settings] = None) -> BlobServiceClient:
    """
    Get Azure Blob Storage service client.
    
    Args:
        settings: Configuration settings
        
    Returns:
        BlobServiceClient instance
    """
    settings = settings or Settings()
    
    try:
        client = BlobServiceClient.from_connection_string(
            settings.azure_storage_connection_string
        )
        logger.info("blob_service_client_created")
        return client
    
    except Exception as e:
        logger.error("failed_to_create_blob_client", error=str(e))
        raise


def upload_report_to_blob(
    local_path: str,
    blob_name: str,
    container_name: str = "drift-reports",
    settings: Optional[Settings] = None
) -> bool:
    """
    Upload drift report to Azure Blob Storage.
    
    Args:
        local_path: Path to local file
        blob_name: Name for blob
        container_name: Container name
        settings: Configuration settings
        
    Returns:
        True if successful, False otherwise
    """
    try:
        client = get_blob_service_client(settings)
        blob_client = client.get_blob_client(
            container=container_name,
            blob=blob_name
        )
        
        with open(local_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        
        logger.info("report_uploaded_to_blob", blob_name=blob_name)
        return True
    
    except Exception as e:
        logger.error("failed_to_upload_report", error=str(e))
        return False


def download_baseline_data(
    blob_name: str,
    local_path: str,
    container_name: str = "baseline-data",
    settings: Optional[Settings] = None
) -> bool:
    """
    Download baseline data from Azure Blob Storage.
    
    Args:
        blob_name: Name of blob to download
        local_path: Local path to save file
        container_name: Container name
        settings: Configuration settings
        
    Returns:
        True if successful, False otherwise
    """
    try:
        client = get_blob_service_client(settings)
        blob_client = client.get_blob_client(
            container=container_name,
            blob=blob_name
        )
        
        # Create local directory if needed
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(local_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        
        logger.info("baseline_data_downloaded", blob_name=blob_name)
        return True
    
    except Exception as e:
        logger.error("failed_to_download_baseline", error=str(e))
        return False


def list_reports(
    container_name: str = "drift-reports",
    settings: Optional[Settings] = None
) -> List[str]:
    """
    List all reports in blob container.
    
    Args:
        container_name: Container name
        settings: Configuration settings
        
    Returns:
        List of blob names
    """
    try:
        client = get_blob_service_client(settings)
        container_client = client.get_container_client(container_name)
        
        blobs = container_client.list_blobs()
        blob_names = [blob.name for blob in blobs]
        
        logger.info("reports_listed", count=len(blob_names))
        return blob_names
    
    except Exception as e:
        logger.error("failed_to_list_reports", error=str(e))
        return []

"""
Azure service integrations.

Handles Azure Storage, Key Vault, SQL Database, and Event Hub operations.
"""

import os
from typing import Optional, Dict, Any
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
from azure.data.tables import TableServiceClient

from src.utils.logger import get_logger

logger = get_logger(__name__)


class AzureKeyVaultService:
    """Azure Key Vault integration for secrets management."""
    
    def __init__(self, vault_url: str):
        """
        Initialize Key Vault client.
        
        Args:
            vault_url (str): Azure Key Vault URL
        """
        self.vault_url = vault_url
        credential = DefaultAzureCredential()
        self.client = SecretClient(vault_url=vault_url, credential=credential)
    
    def get_secret(self, secret_name: str) -> str:
        """
        Retrieve secret from Key Vault.
        
        Args:
            secret_name (str): Name of the secret
            
        Returns:
            str: Secret value
        """
        try:
            secret = self.client.get_secret(secret_name)
            return secret.value
        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_name}: {str(e)}")
            raise


class AzureBlobService:
    """Azure Blob Storage integration."""
    
    def __init__(self, connection_string: str):
        """
        Initialize Blob Storage client.
        
        Args:
            connection_string (str): Azure Storage connection string
        """
        self.client = BlobServiceClient.from_connection_string(connection_string)
    
    def upload_blob(self, container: str, blob_name: str, data: bytes) -> None:
        """
        Upload blob to container.
        
        Args:
            container (str): Container name
            blob_name (str): Blob name
            data (bytes): Data to upload
        """
        try:
            blob_client = self.client.get_blob_client(container=container, blob=blob_name)
            blob_client.upload_blob(data, overwrite=True)
            logger.info(f"Uploaded blob {blob_name} to container {container}")
        except Exception as e:
            logger.error(f"Failed to upload blob: {str(e)}")
            raise
    
    def download_blob(self, container: str, blob_name: str) -> bytes:
        """
        Download blob from container.
        
        Args:
            container (str): Container name
            blob_name (str): Blob name
            
        Returns:
            bytes: Downloaded data
        """
        try:
            blob_client = self.client.get_blob_client(container=container, blob=blob_name)
            return blob_client.download_blob().readall()
        except Exception as e:
            logger.error(f"Failed to download blob: {str(e)}")
            raise
    
    def list_blobs(self, container: str, prefix: str = "") -> list:
        """
        List blobs in container.
        
        Args:
            container (str): Container name
            prefix (str): Blob prefix filter
            
        Returns:
            list: List of blob names
        """
        try:
            container_client = self.client.get_container_client(container)
            return [blob.name for blob in container_client.list_blobs(name_starts_with=prefix)]
        except Exception as e:
            logger.error(f"Failed to list blobs: {str(e)}")
            raise


class AzureTableService:
    """Azure Table Storage integration."""
    
    def __init__(self, connection_string: str):
        """
        Initialize Table Storage client.
        
        Args:
            connection_string (str): Azure Storage connection string
        """
        self.client = TableServiceClient.from_connection_string(connection_string)
    
    def create_table(self, table_name: str) -> None:
        """
        Create table if it doesn't exist.
        
        Args:
            table_name (str): Table name
        """
        try:
            self.client.create_table_if_not_exists(table_name)
            logger.info(f"Table {table_name} created or already exists")
        except Exception as e:
            logger.error(f"Failed to create table: {str(e)}")
            raise
    
    def insert_entity(self, table_name: str, entity: Dict[str, Any]) -> None:
        """
        Insert entity into table.
        
        Args:
            table_name (str): Table name
            entity (Dict): Entity to insert
        """
        try:
            table_client = self.client.get_table_client(table_name)
            table_client.create_entity(entity)
            logger.info(f"Entity inserted into table {table_name}")
        except Exception as e:
            logger.error(f"Failed to insert entity: {str(e)}")
            raise

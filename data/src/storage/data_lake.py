"""
Azure Data Lake service - stores large-scale transaction data
"""

import logging
import json
from typing import Optional, List, BinaryIO
from datetime import datetime

logger = logging.getLogger(__name__)


class DataLakeService:
    """
    Service for storing transaction data in Azure Data Lake (Parquet/Delta format)
    Optimized for big data processing with Spark
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        container_name: str = "transactions"
    ):
        """
        Initialize Data Lake service
        
        Args:
            connection_string: Azure Storage connection string
            container_name: Container name for data storage
        """
        self.connection_string = connection_string
        self.container_name = container_name
        self.client = None
        self.container_client = None
        self._initialized = False

    def connect(self) -> None:
        """Establish connection to Azure Data Lake"""
        try:
            from azure.storage.filedatalake import DataLakeServiceClient

            if not self.connection_string:
                from ..config.settings import settings
                self.connection_string = settings.azure.connection_string

            self.client = DataLakeServiceClient.from_connection_string(
                self.connection_string
            )
            self.container_client = self.client.get_file_system_client(
                file_system=self.container_name
            )
            self._initialized = True
            logger.info(f"Connected to Data Lake container: {self.container_name}")
        except Exception as e:
            logger.error(f"Failed to connect to Data Lake: {str(e)}")
            raise

    def disconnect(self) -> None:
        """Close Data Lake connection"""
        if self.client:
            self.client.close()
            self._initialized = False
            logger.info("Data Lake connection closed")

    def save_parquet(
        self,
        data: "pd.DataFrame",
        path: str,
        partition_by: Optional[List[str]] = None
    ) -> bool:
        """
        Save dataframe as Parquet to Data Lake
        
        Args:
            data: Pandas DataFrame to save
            path: Path in Data Lake (e.g., 'transactions/2025-10-18.parquet')
            partition_by: Optional list of columns to partition by
        
        Returns:
            Success status
        """
        if not self._initialized:
            self.connect()

        try:
            import pandas as pd
            
            # Convert to Parquet bytes
            parquet_bytes = data.to_parquet(index=False)

            # Upload to Data Lake
            file_client = self.container_client.get_file_client(path)
            file_client.upload_data(parquet_bytes, overwrite=True)

            logger.info(f"Saved {len(data)} records to Data Lake: {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save Parquet to Data Lake: {str(e)}")
            return False

    def read_parquet(self, path: str) -> "pd.DataFrame":
        """
        Read Parquet file from Data Lake
        
        Args:
            path: Path in Data Lake
        
        Returns:
            Pandas DataFrame
        """
        if not self._initialized:
            self.connect()

        try:
            import pandas as pd
            import io
            
            # Download from Data Lake
            file_client = self.container_client.get_file_client(path)
            download = file_client.download_file()
            stream = io.BytesIO()
            download.readinto(stream)
            stream.seek(0)

            # Read Parquet
            df = pd.read_parquet(stream)
            logger.info(f"Loaded {len(df)} records from Data Lake: {path}")
            return df

        except Exception as e:
            logger.error(f"Failed to read Parquet from Data Lake: {str(e)}")
            return None

    def save_json_lines(self, data: List[dict], path: str) -> bool:
        """
        Save list of dicts as JSON Lines to Data Lake
        
        Args:
            data: List of dictionaries
            path: Path in Data Lake (e.g., 'transactions/raw/2025-10-18.jsonl')
        
        Returns:
            Success status
        """
        if not self._initialized:
            self.connect()

        try:
            # Convert to JSON Lines format
            jsonl_content = "\n".join([json.dumps(record) for record in data])
            jsonl_bytes = jsonl_content.encode('utf-8')

            # Upload to Data Lake
            file_client = self.container_client.get_file_client(path)
            file_client.upload_data(jsonl_bytes, overwrite=True)

            logger.info(f"Saved {len(data)} records to Data Lake: {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save JSON Lines to Data Lake: {str(e)}")
            return False

    def list_files(self, path: str = "/") -> List[str]:
        """
        List files in Data Lake path
        
        Args:
            path: Path to list (default: root)
        
        Returns:
            List of file paths
        """
        if not self._initialized:
            self.connect()

        try:
            files = []
            paths = self.container_client.get_paths(path=path)
            
            for path_item in paths:
                if not path_item.is_directory:
                    files.append(path_item.name)

            logger.info(f"Found {len(files)} files in {path}")
            return files

        except Exception as e:
            logger.error(f"Failed to list files in Data Lake: {str(e)}")
            return []

    def delete_file(self, path: str) -> bool:
        """
        Delete file from Data Lake
        
        Args:
            path: Path to file
        
        Returns:
            Success status
        """
        if not self._initialized:
            self.connect()

        try:
            file_client = self.container_client.get_file_client(path)
            file_client.delete_file()
            logger.info(f"Deleted file from Data Lake: {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete file from Data Lake: {str(e)}")
            return False

    def get_file_size(self, path: str) -> int:
        """
        Get size of file in Data Lake
        
        Args:
            path: Path to file
        
        Returns:
            File size in bytes
        """
        if not self._initialized:
            self.connect()

        try:
            file_client = self.container_client.get_file_client(path)
            properties = file_client.get_file_properties()
            size = properties['size']
            logger.info(f"File size: {path} = {size} bytes")
            return size

        except Exception as e:
            logger.error(f"Failed to get file size: {str(e)}")
            return 0

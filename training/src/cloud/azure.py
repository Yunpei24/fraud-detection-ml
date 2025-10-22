# training/src/cloud/azure.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    from azure.storage.blob import BlobServiceClient  # type: ignore
    _HAS_BLOB = True
except Exception:
    BlobServiceClient = None  # type: ignore
    _HAS_BLOB = False

try:
    from azure.identity import DefaultAzureCredential  # type: ignore
    from azure.keyvault.secrets import SecretClient  # type: ignore
    _HAS_KEYVAULT = True
except Exception:
    DefaultAzureCredential = None  # type: ignore
    SecretClient = None  # type: ignore
    _HAS_KEYVAULT = False


# -------------------------------------------------------------------
# BLOB STORAGE
# -------------------------------------------------------------------

def _get_blob_client(container: str):
    if not _HAS_BLOB:
        raise RuntimeError("Azure Blob SDK not installed (pip install azure-storage-blob)")
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING environment variable is missing.")
    svc = BlobServiceClient.from_connection_string(conn_str)
    return svc.get_container_client(container)


def upload_to_blob(local_path: str, container: str, blob_name: str) -> bool:
    """
    Upload a local file to Azure Blob Storage.
    """
    try:
        client = _get_blob_client(container)
        with open(local_path, "rb") as data:
            client.upload_blob(name=blob_name, data=data, overwrite=True)
        print(f"[cloud.azure] Uploaded {local_path} → {container}/{blob_name}")
        return True
    except Exception as e:
        print(f"[cloud.azure] Upload failed: {e}")
        return False


def download_from_blob(container: str, blob_name: str, local_path: str) -> bool:
    """
    Download a blob to local path.
    """
    try:
        client = _get_blob_client(container)
        blob_data = client.download_blob(blob_name)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(blob_data.readall())
        print(f"[cloud.azure] Downloaded {container}/{blob_name} → {local_path}")
        return True
    except Exception as e:
        print(f"[cloud.azure] Download failed: {e}")
        return False


# -------------------------------------------------------------------
# KEY VAULT
# -------------------------------------------------------------------

def get_secret_from_vault(vault_url: str, secret_name: str) -> Optional[str]:
    """
    Retrieve a secret (API key, connection string, etc.) from Azure Key Vault.
    Requires `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, and `AZURE_CLIENT_SECRET`
    or managed identity on Databricks.
    """
    if not _HAS_KEYVAULT:
        print("[cloud.azure] KeyVault SDK not installed")
        return None

    try:
        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=vault_url, credential=credential)
        secret = client.get_secret(secret_name)
        print(f"[cloud.azure] Retrieved secret '{secret_name}' from vault.")
        return secret.value
    except Exception as e:
        print(f"[cloud.azure] Failed to fetch secret: {e}")
        return None

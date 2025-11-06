"""
Setup Airflow Connection for Fraud Detection API
=================================================

This script configures the HTTP connection and variables needed for
the canary deployment DAG to communicate with the Fraud Detection API.

**AUTOMATIC TOKEN GENERATION:**
The script automatically obtains an admin JWT token from the API.

Usage (automatic token generation):
    python scripts/setup_api_connection.py \\
        --api-url "https://your-api.azurewebsites.net" \\
        --admin-username "admin" \\
        --admin-password "your-admin-password"
        
Or with explicit token:
    python scripts/setup_api_connection.py \\
        --api-url "https://your-api.azurewebsites.net" \\
        --admin-token "your-jwt-token-here"
        
Or for local development:
    python scripts/setup_api_connection.py \\
        --api-url "http://localhost:8000" \\
        --admin-username "admin" \\
        --admin-password "admin123"
"""

import argparse
import logging
import sys
from typing import Optional, Tuple
from urllib.parse import urlparse

try:
    from airflow.models import Connection, Variable
    from airflow.utils.db import create_session
except ImportError:
    print("ERROR: Airflow not installed. Please install Airflow first.")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("ERROR: requests library not installed. Please install: pip install requests")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_admin_token(api_url: str, username: str, password: str) -> Optional[str]:
    """
    Automatically obtain admin JWT token from the API.

    Args:
        api_url: Full API URL
        username: Admin username
        password: Admin password

    Returns:
        JWT token or None if failed
    """
    try:
        logger.info(f" Obtaining admin token from {api_url}...")

        response = requests.post(
            f"{api_url}/auth/login",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={"username": username, "password": password},
            timeout=10,
        )

        if response.status_code == 200:
            data = response.json()
            token = data.get("access_token")

            if token:
                logger.info(" Admin token obtained successfully")
                logger.info(f"   Token: {token[:20]}...{token[-10:]}")
                logger.info(f"   Expires in: {data.get('expires_in', 'N/A')} seconds")
                return token
            else:
                logger.error(" No access_token in response")
                return None
        else:
            logger.error(f" Login failed: {response.status_code}")
            logger.error(f"   Response: {response.text}")
            return None

    except Exception as e:
        logger.error(f" Failed to obtain token: {e}")
        return None


def create_http_connection(api_url: str, connection_id: str = "fraud_api_connection"):
    """
    Create or update HTTP connection for the API.

    Args:
        api_url: Full API URL (e.g., https://your-api.azurewebsites.net)
        connection_id: Connection ID in Airflow
    """
    try:
        # Parse URL
        parsed = urlparse(api_url)

        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid API URL: {api_url}")

        # Create connection
        conn = Connection(
            conn_id=connection_id,
            conn_type="http",
            host=parsed.netloc,
            schema=parsed.scheme,
            port=parsed.port or (443 if parsed.scheme == "https" else 80),
            description="Fraud Detection API connection for canary deployments",
        )

        # Save or update connection
        with create_session() as session:
            # Check if connection exists
            existing_conn = (
                session.query(Connection).filter_by(conn_id=connection_id).first()
            )

            if existing_conn:
                logger.info(f"Updating existing connection: {connection_id}")
                existing_conn.host = conn.host
                existing_conn.schema = conn.schema
                existing_conn.port = conn.port
                existing_conn.conn_type = conn.conn_type
                existing_conn.description = conn.description
            else:
                logger.info(f"Creating new connection: {connection_id}")
                session.add(conn)

            session.commit()

        logger.info(f" Connection '{connection_id}' configured successfully")
        logger.info(f"   Host: {conn.host}")
        logger.info(f"   Schema: {conn.schema}")
        logger.info(f"   Port: {conn.port}")

        return True

    except Exception as e:
        logger.error(f"Failed to create connection: {e}")
        return False


def set_admin_token(admin_token: str, variable_key: str = "API_ADMIN_TOKEN"):
    """
    Store admin JWT token in Airflow Variables.

    Args:
        admin_token: JWT token for admin authentication
        variable_key: Variable key in Airflow
    """
    try:
        # Set variable
        Variable.set(variable_key, admin_token)

        logger.info(f" Variable '{variable_key}' set successfully")
        logger.info(f"   Token: {admin_token[:20]}...{admin_token[-10:]}")

        return True

    except Exception as e:
        logger.error(f"Failed to set variable: {e}")
        return False


def verify_connection(api_url: str, admin_token: str):
    """
    Verify the connection by calling the deployment status endpoint.

    Args:
        api_url: Full API URL
        admin_token: JWT admin token
    """
    try:
        logger.info(" Verifying connection to API...")

        # Call deployment status endpoint
        response = requests.get(
            f"{api_url}/admin/deployment/deployment-status",
            headers={"Authorization": f"Bearer {admin_token}"},
            timeout=10,
        )

        if response.status_code == 200:
            data = response.json()
            logger.info(" Connection verified successfully!")
            logger.info(f"   Deployment mode: {data.get('deployment_mode')}")
            logger.info(f"   Canary percentage: {data.get('canary_percentage')}%")
            return True
        else:
            logger.error(f" Connection verification failed: {response.status_code}")
            logger.error(f"   Response: {response.text}")
            return False

    except Exception as e:
        logger.error(f" Connection verification failed: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Setup Airflow connection for Fraud Detection API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Automatic token generation (recommended)
  python scripts/setup_api_connection.py \\
    --api-url "https://your-api.azurewebsites.net" \\
    --admin-username "admin" \\
    --admin-password "your-password"
  
  # With explicit token
  python scripts/setup_api_connection.py \\
    --api-url "https://your-api.azurewebsites.net" \\
    --admin-token "eyJhbGc..."
  
  # Local development with auto token
  python scripts/setup_api_connection.py \\
    --api-url "http://localhost:8000" \\
    --admin-username "admin" \\
    --admin-password "admin123" \\
    --verify
        """,
    )
    parser.add_argument(
        "--api-url",
        required=True,
        help="Full API URL (e.g., https://your-api.azurewebsites.net or http://localhost:8000)",
    )

    # Token generation options
    token_group = parser.add_mutually_exclusive_group(required=True)
    token_group.add_argument("--admin-token", help="JWT admin token (explicit token)")
    token_group.add_argument(
        "--admin-username", help="Admin username (for automatic token generation)"
    )

    parser.add_argument(
        "--admin-password", help="Admin password (required with --admin-username)"
    )
    parser.add_argument(
        "--connection-id",
        default="fraud_api_connection",
        help="Connection ID in Airflow (default: fraud_api_connection)",
    )
    parser.add_argument(
        "--variable-key",
        default="API_ADMIN_TOKEN",
        help="Variable key for admin token (default: API_ADMIN_TOKEN)",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify connection after setup"
    )
    parser.add_argument(
        "--auto-renew",
        action="store_true",
        help="Enable automatic token renewal (stores credentials securely)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.admin_username and not args.admin_password:
        parser.error("--admin-password is required when using --admin-username")

    # Validate URL
    api_url = args.api_url.rstrip("/")

    logger.info(" Setting up Airflow connection for Fraud Detection API")
    logger.info(f"   API URL: {api_url}")

    # Get or generate admin token
    admin_token = args.admin_token

    if args.admin_username:
        logger.info(f"   Mode: Automatic token generation")
        logger.info(f"   Username: {args.admin_username}")

        admin_token = get_admin_token(api_url, args.admin_username, args.admin_password)

        if not admin_token:
            logger.error(" Failed to obtain admin token")
            logger.error("   Check your username, password, and API URL")
            sys.exit(1)

        # Store credentials for auto-renewal if requested
        if args.auto_renew:
            logger.info(" Storing credentials for automatic token renewal...")
            Variable.set("API_ADMIN_USERNAME", args.admin_username)
            Variable.set(
                "API_ADMIN_PASSWORD", args.admin_password, serialize_json=False
            )
            logger.info(" Credentials stored securely in Airflow Variables")
    else:
        logger.info(f"   Mode: Explicit token")

    # Create connection
    if not create_http_connection(api_url, args.connection_id):
        logger.error(" Failed to create connection")
        sys.exit(1)

    # Store API URL for token renewal DAG
    Variable.set("API_URL", api_url)
    logger.info(f" API URL stored in variables")

    # Set admin token
    if not set_admin_token(admin_token, args.variable_key):
        logger.error(" Failed to set admin token")
        sys.exit(1)

    # Verify connection if requested
    if args.verify:
        if not verify_connection(api_url, admin_token):
            logger.warning(
                " Connection verification failed, but configuration is saved"
            )
            logger.warning("   Check API URL and admin credentials")

    logger.info("")
    logger.info(" Setup complete!")
    logger.info("")
    logger.info(" Next steps:")
    logger.info("   1. Verify the connection in Airflow UI: Admin > Connections")
    logger.info("   2. Check the variable in Airflow UI: Admin > Variables")
    logger.info("   3. Trigger the DAG: dags/05_model_deployment_canary_http")
    logger.info("")

    if args.auto_renew:
        logger.info(" Automatic token renewal enabled:")
        logger.info("   • Credentials stored in Airflow Variables")
        logger.info("   • Create a DAG to renew token periodically (daily recommended)")
        logger.info("")

    logger.info(" To update connection or renew token, run this script again")

    sys.exit(0)


if __name__ == "__main__":
    main()

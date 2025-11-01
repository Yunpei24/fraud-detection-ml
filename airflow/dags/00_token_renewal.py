"""
DAG 00: Automatic Admin Token Renewal
======================================

Automatically renews the admin JWT token for API authentication.

This DAG:
1. Reads admin credentials from Airflow Variables
2. Obtains a new JWT token from the API
3. Updates the API_ADMIN_TOKEN variable
4. Runs daily to ensure token is always valid

Schedule: Daily at 2:00 AM
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pendulum
import requests
from airflow.models.dag import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

logger = logging.getLogger(__name__)

# Default args
default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email": ["ml-alerts@frauddetection.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}


def renew_admin_token(**context):
    """
    Renew admin JWT token from the API.

    This function:
    1. Reads credentials from Airflow Variables
    2. Calls /auth/login to get new token
    3. Updates API_ADMIN_TOKEN variable

    Returns:
        Dict with renewal status
    """
    try:
        # Get API URL from connection (stored during setup)
        api_url = Variable.get("API_URL", default_var=None)

        if not api_url:
            logger.error("❌ API_URL variable not found")
            logger.error("   Run setup_api_connection.py first")
            raise ValueError("API_URL not configured")

        # Get admin credentials
        admin_username = Variable.get("API_ADMIN_USERNAME", default_var=None)
        admin_password = Variable.get("API_ADMIN_PASSWORD", default_var=None)

        if not admin_username or not admin_password:
            logger.error("❌ Admin credentials not found in Airflow Variables")
            logger.error("   Run setup_api_connection.py with --auto-renew flag")
            raise ValueError("Admin credentials not configured")

        logger.info(f"🔐 Renewing admin token from {api_url}...")
        logger.info(f"   Username: {admin_username}")

        # Call login endpoint
        response = requests.post(
            f"{api_url}/auth/login",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={"username": admin_username, "password": admin_password},
            timeout=10,
        )

        if response.status_code == 200:
            data = response.json()
            new_token = data.get("access_token")

            if new_token:
                # Update token in Airflow Variables
                Variable.set("API_ADMIN_TOKEN", new_token)

                logger.info("✅ Admin token renewed successfully")
                logger.info(f"   New token: {new_token[:20]}...{new_token[-10:]}")
                logger.info(f"   Expires in: {data.get('expires_in', 'N/A')} seconds")

                # Log to XCom for monitoring
                context["task_instance"].xcom_push(
                    key="token_renewed",
                    value={
                        "status": "success",
                        "timestamp": datetime.utcnow().isoformat(),
                        "expires_in": data.get("expires_in"),
                        "token_preview": f"{new_token[:20]}...{new_token[-10:]}",
                    },
                )

                return {
                    "status": "success",
                    "message": "Token renewed successfully",
                    "expires_in": data.get("expires_in"),
                }
            else:
                logger.error("❌ No access_token in response")
                raise ValueError("No access_token in API response")
        else:
            logger.error(f"❌ Token renewal failed: {response.status_code}")
            logger.error(f"   Response: {response.text}")
            raise ValueError(f"API returned status {response.status_code}")

    except Exception as e:
        logger.error(f"❌ Token renewal failed: {e}")
        raise


def verify_new_token(**context):
    """
    Verify the new token works by calling a protected endpoint.

    Returns:
        Dict with verification status
    """
    try:
        api_url = Variable.get("API_URL")
        new_token = Variable.get("API_ADMIN_TOKEN")

        logger.info("🔍 Verifying new token...")

        # Call deployment status endpoint
        response = requests.get(
            f"{api_url}/admin/deployment/deployment-status",
            headers={"Authorization": f"Bearer {new_token}"},
            timeout=10,
        )

        if response.status_code == 200:
            logger.info("✅ New token verified successfully")
            logger.info(f"   API is accessible with new token")

            return {
                "status": "success",
                "message": "Token verified",
                "api_status": "accessible",
            }
        else:
            logger.error(f"❌ Token verification failed: {response.status_code}")
            raise ValueError(f"API returned status {response.status_code}")

    except Exception as e:
        logger.error(f"❌ Token verification failed: {e}")
        raise


# Create DAG
with DAG(
    dag_id="00_token_renewal",
    default_args=default_args,
    description="Automatically renew admin JWT token for API authentication",
    schedule="0 2 * * *",  # Daily at 2:00 AM
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    tags=["maintenance", "authentication", "security"],
    max_active_runs=1,
) as dag:

    # Task documentation
    dag.doc_md = __doc__

    # Task 1: Renew token
    renew_token = PythonOperator(
        task_id="renew_admin_token",
        python_callable=renew_admin_token,
        provide_context=True,
        doc_md="""
        ## Renew Admin Token
        
        Obtains a new JWT token from the API using stored credentials.
        
        **Prerequisites:**
        - API_URL variable must be set
        - API_ADMIN_USERNAME variable must be set
        - API_ADMIN_PASSWORD variable must be set
        
        **Output:**
        - Updates API_ADMIN_TOKEN variable
        - Logs token renewal status
        """,
    )

    # Task 2: Verify token
    verify_token = PythonOperator(
        task_id="verify_new_token",
        python_callable=verify_new_token,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_SUCCESS,
        doc_md="""
        ## Verify New Token
        
        Tests the new token by calling a protected API endpoint.
        
        **Verification:**
        - Calls /admin/deployment/deployment-status
        - Ensures token has admin permissions
        - Confirms API accessibility
        """,
    )

    # Task flow
    renew_token >> verify_token

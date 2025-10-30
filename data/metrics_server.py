#!/usr/bin/env python3
"""
Standalone Prometheus metrics server for data pipeline monitoring.

This script starts an HTTP server to expose metrics to Prometheus.
The port can be configured via PROMETHEUS_PORT environment variable (default: 9091).
The server runs continuously until manually stopped.

Usage:
    python metrics_server.py
    PROMETHEUS_PORT=9091 python metrics_server.py

The metrics will be available at: http://localhost:<port>/metrics
"""

import logging
import os
import signal
import sys
import time

from src.monitoring.metrics import setup_prometheus_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
running = True

# Get port from environment variable or use default
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "9091"))


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    global running
    logger.info(f"Shutdown signal received: {sig}")
    running = False
    sys.exit(0)


def main():
    """Start the Prometheus metrics server."""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info(f"Starting data pipeline metrics server on port {PROMETHEUS_PORT}")

    try:
        # Start Prometheus HTTP server
        setup_prometheus_metrics(port=PROMETHEUS_PORT)
        logger.info(
            f"✅ Metrics server started successfully at http://localhost:{PROMETHEUS_PORT}/metrics"
        )

        # Keep the server running
        while running:
            time.sleep(60)  # Sleep for 1 minute

    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(
                f"❌ Port {PROMETHEUS_PORT} already in use. Another process is using this port."
            )
            sys.exit(1)
        else:
            logger.error(f"❌ Server startup failed: {e}", exc_info=True)
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

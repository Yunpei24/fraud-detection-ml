#!/usr/bin/env python3
"""
Standalone Prometheus metrics server for drift detection monitoring.

This script starts an HTTP server on port 9095 to expose metrics to Prometheus.
The server runs continuously until manually stopped.

Usage:
    python metrics_server.py

The metrics will be available at: http://localhost:9095/metrics
"""

import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import structlog
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, generate_latest

# Import drift metrics to register them in the global REGISTRY
from src.monitoring.metrics import (
    alert_counter,
    drift_score_gauge,
    fraud_rate_gauge,
    model_recall_gauge,
)

logger = structlog.get_logger(__name__)

# Global flag for graceful shutdown
running = True


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    global running
    logger.info("shutdown_signal_received", signal=sig)
    running = False
    sys.exit(0)


class MetricsHandler(BaseHTTPRequestHandler):
    """Custom HTTP request handler for metrics and health endpoints."""

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", CONTENT_TYPE_LATEST)
            self.end_headers()
            # Use global REGISTRY to include all drift metrics
            self.wfile.write(generate_latest(REGISTRY))
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                b'{"status": "healthy", "service": "drift-detection-metrics"}'
            )
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def log_message(self, format, *args):
        """Override to use structlog instead of default logging."""
        logger.debug("http_request", message=format % args)


def start_custom_metrics_server(port: int) -> HTTPServer:
    """
    Start a custom HTTP server that serves both Prometheus metrics and health checks.

    Args:
        port: Port to start the server on

    Returns:
        The HTTP server instance
    """
    # Log registered metrics for debugging
    logger.info(
        "registered_drift_metrics",
        collectors=[
            collector.__class__.__name__
            for collector in REGISTRY._collector_to_names.keys()
        ],
    )

    # Start the custom HTTP server
    server = HTTPServer(("0.0.0.0", port), MetricsHandler)
    logger.info("custom_metrics_server_started", port=port)
    return server


def start_metrics_server_with_retry(port: int, max_retries: int = 5) -> bool:
    """
    Start Prometheus metrics server with exponential backoff retry logic.

    Args:
        port: Port to start the server on
        max_retries: Maximum number of retry attempts

    Returns:
        True if server started successfully, False otherwise
    """
    for attempt in range(max_retries + 1):
        try:
            logger.info(
                "attempting_to_start_metrics_server", attempt=attempt + 1, port=port
            )

            # Try to start the custom server
            server = start_custom_metrics_server(port)

            # Start server in a separate thread
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()

            logger.info(
                "metrics_server_started_successfully",
                endpoint=f"http://localhost:{port}/metrics",
                health_endpoint=f"http://localhost:{port}/health",
            )
            return True

        except OSError as e:
            if "Address already in use" in str(e):
                if attempt < max_retries:
                    wait_time = (
                        2**attempt
                    )  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                    logger.warning(
                        "port_already_in_use",
                        port=port,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        wait_time_seconds=wait_time,
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(
                        "port_already_in_use_after_retries",
                        port=port,
                        max_retries=max_retries,
                    )
                    return False
            else:
                logger.error("server_startup_failed", error=str(e), exc_info=True)
                return False
        except Exception as e:
            logger.error("unexpected_error_during_startup", error=str(e), exc_info=True)
            return False

    return False


def main():
    """Start the Prometheus metrics server."""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    port = int(os.getenv("PROMETHEUS_PORT", 9095))

    logger.info("starting_drift_detection_metrics_server", port=port)

    # Start server with retry logic
    if not start_metrics_server_with_retry(port):
        logger.error("failed_to_start_metrics_server_after_retries")
        sys.exit(1)

    # Keep the server running
    while running:
        time.sleep(60)  # Sleep for 1 minute


if __name__ == "__main__":
    main()

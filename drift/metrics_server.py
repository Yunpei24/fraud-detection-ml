#!/usr/bin/env python3
"""
Standalone Prometheus metrics server for drift detection monitoring.

This script starts an HTTP server on port 9091 to expose metrics to Prometheus.
The server runs continuously until manually stopped.

Usage:
    python metrics_server.py

The metrics will be available at: http://localhost:9091/metrics
"""

import time
import signal
import sys
from src.monitoring.metrics import setup_prometheus_metrics
import structlog

logger = structlog.get_logger(__name__)

# Global flag for graceful shutdown
running = True


def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    global running
    logger.info("shutdown_signal_received", signal=sig)
    running = False
    sys.exit(0)


def main():
    """Start the Prometheus metrics server."""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("starting_drift_detection_metrics_server", port=9091)
    
    try:
        # Start Prometheus HTTP server
        setup_prometheus_metrics(port=9091)
        logger.info("metrics_server_started_successfully", 
                   endpoint="http://localhost:9091/metrics")
        
        # Keep the server running
        while running:
            time.sleep(60)  # Sleep for 1 minute
            
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error("port_already_in_use", port=9091, 
                        error="Another process is using port 9091")
            sys.exit(1)
        else:
            logger.error("server_startup_failed", error=str(e), exc_info=True)
            sys.exit(1)
    except Exception as e:
        logger.error("unexpected_error", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

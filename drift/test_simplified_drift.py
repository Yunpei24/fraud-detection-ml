#!/usr/bin/env python3
"""
Test script for the simplified drift component.

This script tests the basic functionality of the refactored drift component
that uses the API drift service instead of custom detection logic.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import required modules
from config.settings import Settings
from api_client import FraudDetectionAPIClient

def test_api_client():
    """Test that the API client can be instantiated."""
    print("Testing API client instantiation...")

    try:
        from api_client import FraudDetectionAPIClient
        from config.settings import Settings

        # Test with default settings
        settings = Settings()
        client = FraudDetectionAPIClient(
            base_url=getattr(settings, 'api_base_url', 'http://localhost:8000'),
            timeout=getattr(settings, 'api_timeout', 30)
        )

        print(f"✅ API client created with base_url: {client.base_url}")
        return True
    except Exception as e:
        print(f"❌ API client test failed: {e}")
        return False

def test_settings():
    """Test that settings are properly configured."""
    print("Testing settings configuration...")

    try:
        from config.settings import Settings

        settings = Settings()

        # Check essential settings
        required_attrs = [
            'database_url',
            'api_base_url',
            'api_timeout',
            'data_drift_threshold',
            'target_drift_threshold',
            'concept_drift_threshold',
            'retraining_cooldown_hours',
            'alert_email_enabled',
            'prometheus_enabled',
            'airflow_api_url'
        ]

        for attr in required_attrs:
            if hasattr(settings, attr):
                value = getattr(settings, attr)
                print(f"✅ {attr}: {value}")
            else:
                print(f"❌ Missing required setting: {attr}")
                return False

        return True
    except Exception as e:
        print(f"❌ Settings test failed: {e}")
        return False

def test_pipeline_import():
    """Test that the pipeline can be imported."""
    print("Testing pipeline import...")

    try:
        from pipelines.hourly_monitoring import (
            call_api_drift_detection,
            check_thresholds,
            trigger_alerts,
            run_hourly_monitoring
        )
        print("✅ Pipeline functions imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Pipeline import failed: {e}")
        return False

def test_retraining_trigger():
    """Test that retraining trigger works with API results."""
    print("Testing retraining trigger...")

    try:
        from retraining.trigger import RetrainingTrigger

        trigger = RetrainingTrigger()

        # Test with mock API results
        mock_api_results = {
            "drift_summary": {
                "overall_drift_detected": True,
                "severity_score": 2.5,
                "drift_types_detected": ["data_drift"]
            },
            "data_drift": {"drift_detected": True},
            "target_drift": {"drift_detected": False},
            "concept_drift": {"drift_detected": False}
        }

        should_retrain, reason = trigger.should_retrain(mock_api_results)
        print(f"✅ Retraining decision: {should_retrain} (reason: {reason})")

        priority = trigger.get_retrain_priority(mock_api_results)
        print(f"✅ Retraining priority: {priority}")

        return True
    except Exception as e:
        print(f"❌ Retraining trigger test failed: {e}")
        return False

def test_api_client():
    """Test that the API client can be instantiated."""
    print("Testing API client instantiation...")

    # Test with default settings
    settings = Settings()
    client = FraudDetectionAPIClient(
        base_url=getattr(settings, 'api_base_url', 'http://localhost:8000'),
        timeout=getattr(settings, 'api_timeout', 30)
    )

    print(f"✅ API client created with base_url: {client.base_url}")
    return True

def test_settings():
    """Test that settings are properly configured."""
    print("Testing settings configuration...")

    settings = Settings()

    # Check essential settings
    required_attrs = [
        'database_url',
        'api_base_url',
        'api_timeout',
        'data_drift_threshold',
        'target_drift_threshold',
        'concept_drift_threshold',
        'retraining_cooldown_hours',
        'alert_email_enabled',
        'prometheus_enabled',
        'airflow_api_url'
    ]

    for attr in required_attrs:
        if hasattr(settings, attr):
            value = getattr(settings, attr)
            print(f"✅ {attr}: {value}")
        else:
            print(f"❌ Missing required setting: {attr}")
            return False

    return True

def test_pipeline_import():
    """Test that the pipeline can be imported."""
    print("Testing pipeline import...")

    try:
        from pipelines.hourly_monitoring import (
            call_api_drift_detection,
            check_thresholds,
            trigger_alerts,
            run_hourly_monitoring
        )
        print("✅ Pipeline functions imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Pipeline import failed: {e}")
        return False

def test_retraining_trigger():
    """Test that retraining trigger works with API results."""
    print("Testing retraining trigger...")

    try:
        from retraining.trigger import RetrainingTrigger

        trigger = RetrainingTrigger()

        # Test with mock API results
        mock_api_results = {
            "drift_summary": {
                "overall_drift_detected": True,
                "severity_score": 2.5,
                "drift_types_detected": ["data_drift"]
            },
            "data_drift": {"drift_detected": True},
            "target_drift": {"drift_detected": False},
            "concept_drift": {"drift_detected": False}
        }

        should_retrain, reason = trigger.should_retrain(mock_api_results)
        print(f"✅ Retraining decision: {should_retrain} (reason: {reason})")

        priority = trigger.get_retrain_priority(mock_api_results)
        print(f"✅ Retraining priority: {priority}")

        return True
    except Exception as e:
        print(f"❌ Retraining trigger test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Simplified Drift Component")
    print("=" * 50)

    tests = [
        test_api_client,
        test_settings,
        test_pipeline_import,
        test_retraining_trigger
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            print()

    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All tests passed! The simplified drift component is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
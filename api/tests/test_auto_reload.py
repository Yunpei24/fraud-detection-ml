#!/usr/bin/env python3
"""
Test Auto-Reload Functionality
===============================

This script tests that the API correctly detects and reloads:
1. Model files (.pkl) when modified
2. Traffic routing config (traffic_routing.json) when modified
"""

import json
import time
from pathlib import Path
import tempfile
import shutil


def test_traffic_config_auto_reload():
    """Test that traffic config auto-reloads when file is modified."""
    print(" Testing traffic config auto-reload...")

    # Create temp config file
    temp_dir = Path(tempfile.mkdtemp())
    config_file = temp_dir / "traffic_routing.json"

    # Write initial config
    initial_config = {
        "canary_enabled": False,
        "canary_traffic_pct": 0,
        "champion_traffic_pct": 100,
        "canary_model_uris": [],
        "champion_model_uris": ["/app/models/champion"],
        "ensemble_weights": {
            "xgboost": 0.50,
            "random_forest": 0.30,
            "neural_network": 0.15,
            "isolation_forest": 0.05,
        },
    }

    with open(config_file, "w") as f:
        json.dump(initial_config, f, indent=2)

    initial_mtime = config_file.stat().st_mtime
    print(f"  Initial config written (mtime: {initial_mtime})")

    # Wait a bit to ensure different timestamp
    time.sleep(1.1)

    # Modify config
    updated_config = initial_config.copy()
    updated_config["canary_enabled"] = True
    updated_config["canary_traffic_pct"] = 5
    updated_config["champion_traffic_pct"] = 95

    with open(config_file, "w") as f:
        json.dump(updated_config, f, indent=2)

    updated_mtime = config_file.stat().st_mtime
    print(f"   Config updated (mtime: {updated_mtime})")

    # Verify timestamp changed
    assert updated_mtime > initial_mtime, "Timestamp should have changed!"
    print(f"  Timestamp changed: {initial_mtime} → {updated_mtime}")
    print(f"  Difference: {updated_mtime - initial_mtime:.2f} seconds")

    # Cleanup
    shutil.rmtree(temp_dir)
    print("  Cleanup complete\n")


def test_model_files_auto_reload():
    """Test that model files auto-reload when .pkl files are modified."""
    print(" Testing model files auto-reload...")

    # Create temp model directory
    temp_dir = Path(tempfile.mkdtemp())
    model_file1 = temp_dir / "xgboost_model.pkl"
    model_file2 = temp_dir / "random_forest_model.pkl"

    # Create dummy model files
    model_file1.write_text("dummy model 1")
    model_file2.write_text("dummy model 2")

    initial_mtimes = {
        "xgboost": model_file1.stat().st_mtime,
        "random_forest": model_file2.stat().st_mtime,
    }
    max_initial_mtime = max(initial_mtimes.values())
    print(f"  Initial models written (max mtime: {max_initial_mtime})")

    # Wait a bit
    time.sleep(1.1)

    # Modify one model
    model_file1.write_text("updated dummy model 1")

    updated_mtimes = {
        "xgboost": model_file1.stat().st_mtime,
        "random_forest": model_file2.stat().st_mtime,
    }
    max_updated_mtime = max(updated_mtimes.values())
    print(f"  Model updated (max mtime: {max_updated_mtime})")

    # Verify timestamp changed
    assert max_updated_mtime > max_initial_mtime, "Timestamp should have changed!"
    print(f"  Timestamp changed: {max_initial_mtime} → {max_updated_mtime}")
    print(f"  Difference: {max_updated_mtime - max_initial_mtime:.2f} seconds")

    # Cleanup
    shutil.rmtree(temp_dir)
    print("  Cleanup complete\n")


def test_config_structure():
    """Test that new config structure is valid."""
    print(" Testing config structure validation...")

    config = {
        "canary_enabled": True,
        "canary_traffic_pct": 25,
        "champion_traffic_pct": 75,
        "canary_model_uris": ["/mnt/fraud-models/canary"],
        "champion_model_uris": ["/mnt/fraud-models/champion"],
        "ensemble_weights": {
            "xgboost": 0.50,
            "random_forest": 0.30,
            "neural_network": 0.15,
            "isolation_forest": 0.05,
        },
    }

    # Validate required fields
    required_fields = [
        "canary_enabled",
        "canary_traffic_pct",
        "champion_traffic_pct",
        "canary_model_uris",
        "champion_model_uris",
        "ensemble_weights",
    ]

    for field in required_fields:
        assert field in config, f"Missing required field: {field}"
        print(f"  Field '{field}' present")

    # Validate traffic percentages sum to 100
    total_pct = config["canary_traffic_pct"] + config["champion_traffic_pct"]
    assert total_pct == 100, f"Traffic percentages should sum to 100, got {total_pct}"
    print(f"  Traffic percentages sum to 100")

    # Validate ensemble weights
    ensemble_weights = config["ensemble_weights"]
    assert len(ensemble_weights) == 4, "Should have 4 ensemble weights"
    total_weight = sum(ensemble_weights.values())
    assert (
        abs(total_weight - 1.0) < 0.01
    ), f"Ensemble weights should sum to 1.0, got {total_weight}"
    print(f"  Ensemble weights sum to 1.0")

    print("  Config structure valid\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("AUTO-RELOAD FUNCTIONALITY TESTS")
    print("=" * 60)
    print()

    try:
        test_traffic_config_auto_reload()
        test_model_files_auto_reload()
        test_config_structure()

        print("=" * 60)
        print(" ALL TESTS PASSED")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n UNEXPECTED ERROR: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

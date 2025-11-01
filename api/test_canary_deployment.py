#!/usr/bin/env python3
"""
Test Canary Deployment Flow
===========================
Test the complete canary deployment workflow.

This script:
1. Simulates canary deployment with 25% traffic
2. Tests traffic routing
3. Simulates promotion to production
4. Verifies traffic routing updates

Usage:
    python test_canary_deployment.py
"""
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_canary_deployment_flow():
    """Test the complete canary deployment flow."""
    logger.info("🧪 Testing canary deployment flow...")

    # Create temporary directories for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Set up test directories
        api_dir = temp_path / "api"
        models_dir = api_dir / "models"
        config_dir = api_dir / "config"
        champion_dir = models_dir / "champion"
        canary_dir = models_dir / "canary"

        models_dir.mkdir(parents=True)
        config_dir.mkdir(parents=True)
        champion_dir.mkdir()
        canary_dir.mkdir()

        # Create mock champion models
        logger.info("📦 Creating mock champion models...")
        (champion_dir / "xgboost_model.pkl").write_text("mock xgboost model")
        (champion_dir / "random_forest_model.pkl").write_text("mock rf model")
        (champion_dir / "nn_model.pth").write_text("mock nn model")
        (champion_dir / "isolation_forest_model.pkl").write_text("mock iforest model")

        # Step 1: Simulate canary deployment (25% traffic)
        logger.info("🚀 Step 1: Simulating canary deployment (25% traffic)...")

        # Create mock canary models
        (canary_dir / "xgboost_model.pkl").write_text("mock canary xgboost model")
        (canary_dir / "random_forest_model.pkl").write_text("mock canary rf model")
        (canary_dir / "nn_model.pth").write_text("mock canary nn model")
        (canary_dir / "isolation_forest_model.pkl").write_text(
            "mock canary iforest model"
        )

        # Create traffic routing config for 25% canary
        config_file = config_dir / "traffic_routing.json"
        config = {
            "canary_percentage": 25,
            "canary_model_path": str(canary_dir),
            "champion_model_path": str(champion_dir),
        }

        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        logger.info("✅ Created canary deployment config")

        # Step 2: Test traffic routing
        logger.info("🔀 Step 2: Testing traffic routing...")

        # Import and test TrafficRouter
        sys.path.insert(0, str(api_dir))
        os.environ["TRAFFIC_ROUTING_CONFIG"] = str(config_file)

        from src.services.traffic_router import TrafficRouter

        router = TrafficRouter()
        logger.info(f"   Canary percentage: {router.canary_percentage}")
        logger.info(f"   Champion path: {router.champion_model_path}")
        logger.info(f"   Canary path: {router.canary_model_path}")

        # Test routing decisions
        canary_count = 0
        total_tests = 1000

        for _ in range(total_tests):
            if router.should_use_canary():
                canary_count += 1

        canary_rate = canary_count / total_tests
        logger.info(
            f"   Traffic routing test: {canary_rate:.1%} canary (expected ~25%)"
        )

        # Verify routing is approximately correct
        assert (
            0.20 <= canary_rate <= 0.30
        ), f"Traffic routing incorrect: {canary_rate:.1%}"

        # Step 3: Simulate promotion to production
        logger.info("🎯 Step 3: Simulating promotion to production...")

        # Move canary models to champion (as done in promote_to_production.py)
        shutil.rmtree(champion_dir)
        shutil.move(str(canary_dir), str(champion_dir))

        # Update config for 0% canary
        config = {
            "canary_percentage": 0,
            "canary_model_path": str(canary_dir),
            "champion_model_path": str(champion_dir),
        }

        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

        logger.info("✅ Promoted canary models to champion")

        # Step 4: Test traffic routing after promotion
        logger.info("🔄 Step 4: Testing traffic routing after promotion...")

        # Reload config
        router.reload_config()
        logger.info(f"   New canary percentage: {router.canary_percentage}")

        # Test routing decisions (should be 0% canary now)
        canary_count = 0
        for _ in range(100):
            if router.should_use_canary():
                canary_count += 1

        logger.info(f"   Post-promotion routing: {canary_count}% canary (expected 0%)")

        # Verify no canary traffic after promotion
        assert (
            canary_count == 0
        ), f"Traffic routing incorrect after promotion: {canary_count}% canary"

        # Step 5: Verify model files
        logger.info("📁 Step 5: Verifying model files...")

        # Champion directory should have the promoted models
        champion_files = list(champion_dir.glob("*"))
        logger.info(f"   Champion models: {[f.name for f in champion_files]}")

        # Canary directory should be empty or gone
        if canary_dir.exists():
            canary_files = list(canary_dir.glob("*"))
            logger.info(f"   Canary models: {[f.name for f in canary_files]}")
        else:
            logger.info("   Canary directory removed")

        # Verify champion models exist
        expected_files = [
            "xgboost_model.pkl",
            "random_forest_model.pkl",
            "nn_model.pth",
            "isolation_forest_model.pkl",
        ]
        actual_files = [f.name for f in champion_files]
        assert set(expected_files) == set(
            actual_files
        ), f"Champion models incorrect: {actual_files}"

        logger.info("🎉 Canary deployment flow test completed successfully!")
        return True


def main():
    """Main entry point."""
    try:
        success = test_canary_deployment_flow()
        if success:
            logger.info("✅ All tests passed!")
            sys.exit(0)
        else:
            logger.error("❌ Tests failed!")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

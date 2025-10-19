import os
def test_processed_exists():
    assert os.path.exists("data/processed/train.parquet")

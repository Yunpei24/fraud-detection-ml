# training/src/pipelines/__init__.py
from .training_pipeline import run_training
from .comparison_pipeline import run_comparison

__all__ = ["run_training", "run_comparison"]

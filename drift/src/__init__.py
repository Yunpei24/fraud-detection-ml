"""
Drift Detection Component

Monitors production ML system for drift:
- Data Drift: Feature distribution changes
- Target Drift: Label distribution changes  
- Concept Drift: Model performance degradation

This component integrates with:
- API component (for predictions)
- Data component (for historical data)
- Training component (for retraining triggers)
"""

__version__ = "1.0.0"
__author__ = "Fraud Detection ML Team"

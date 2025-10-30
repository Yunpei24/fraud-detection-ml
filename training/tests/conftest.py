# training/tests/conftest.py
import pytest
import numpy as np
import pandas as pd

@pytest.fixture(scope="session")
def tiny_credit_df():
    """ A synthetic, tiny dataframe to use in tests """
    n = 80
    rng = np.random.default_rng(123)
    data = {f"v{i}": rng.normal(0, 1, n) for i in range(1, 29)}
    data["Time"] = np.arange(n, dtype=float) * 2.0
    data["amount"] = np.abs(rng.normal(50, 25, n))
    y = np.zeros(n, dtype=int)
    y[:3] = 1
    rng.shuffle(y)
    data["Class"] = y
    return pd.DataFrame(data)

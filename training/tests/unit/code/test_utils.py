import sys
import os
import numpy as np

# Correct sys.path to point to the 'training/src' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'training', 'src')))

from training.src.utils.helpers import set_seed  # Correct import

def test_set_seed_idempotent():
    set_seed(42)
    a = np.random.rand(3)
    set_seed(42)
    b = np.random.rand(3)
    set_seed(42)
    c = np.random.rand(3)
    assert np.allclose(a, b) and np.allclose(b, c)

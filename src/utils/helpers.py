def set_seed(seed: int = 42):
    import numpy as np
    import random
    import os
    import torch as _torch  
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

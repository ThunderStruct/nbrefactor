import torch
import numpy as np

def set_reproducible(random_seed):
    # torch
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # numpy
    np.random.seed(random_seed)



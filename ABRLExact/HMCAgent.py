import numpy as np

import torch

import pyro
import pyro.distributions as ndist
from pyro.infer import HMC, NUTS



def get_default_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu")


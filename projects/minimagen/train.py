# Our imports
import utils

# Native imports
from typing import Tuple, List, Dict, Optional, Union, Callable

# Third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import IPython
from tqdm import tqdm
from minimagen.t5 import t5_encode_text

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



m = GaussianDiffusion(timesteps=1000)

x_start = torch.randn(1, 3, 32, 32)
t = torch.tensor([5])
_ = m.q_sample(x_start, t)
IPython.embed()
#
import time
import tqdm
import torch
from dataclasses import dataclass


@dataclass
class ParallelConfig:
    # Global tp degree  [-1, 1, 2, 4, 8, 16]
    tp_degree: int = -1
    # Global tp consecutive [-1]
    tp_consecutive: int = -1
    # Pipeline parallel degree, [1, 2, 4, 8, 16]
    pp_degree: int = 2
    # local bs
    bs_local: int = 32
    # model layers
    model_num_layers: int = 48

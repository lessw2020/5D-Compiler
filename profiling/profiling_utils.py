# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
# code based on: https://github.com/Hsword/Hetu/tree/main/tools/Galvatron

import torch
import torch.nn as nn
import torch.distributed as dist

import argparse
import os
import time

import numpy as np
import random


class RankPrint:
    def __init__(self, rank, rank_to_print: int = 0):
        self.rank = rank
        self.rank_to_print = rank_to_print
        self.maybe_print = self.rank == self.rank_to_print

    def __call__(self, msg):
        if self.maybe_print:
            print(f"{msg}")


def setup():
    """we use torchrun for init so no params needed here"""
    dist.init_process_group("nccl")


def setup_environ_flags(rank=0):
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    # if cfg.nccl_debug_handler:
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # if cfg.distributed_debug:
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    if rank == 0:
        print(f"--> running with torch dist debug set to detail")


def clear_gpu_cache(rank=None):
    if rank == 0:
        print(f"clearing gpu cache for all ranks")
    torch.cuda.empty_cache()


def cleanup():
    dist.barrier()
    dist.destroy_process_group()


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

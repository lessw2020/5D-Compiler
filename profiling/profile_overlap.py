# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.

import torch
import torch.nn as nn
import torch.distributed as dist

import argparse
import os
import time

from torch.utils.data import DistributedSampler
import contextlib

# add DDP support
from torch.nn.parallel import DistributedDataParallel as DDP

# import colorama
# from colorama import Fore


from tqdm import tqdm
import numpy as np
import random

from torch.nn.parallel import DistributedDataParallel as DDP
import os, sys

sys.path.insert(0, "..")
sys.path.insert(0, "../..")
# from utils import read_json_config, write_json_config

# _none_context = contextlib.nullcontext()


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def setup():
    """we use torchrun for init so no params needed here"""
    dist.init_process_group("nccl")


def setup_environ_flags(cfg, rank=0):
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    if cfg.nccl_debug_handler:
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    if cfg.distributed_debug:
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


def setup_tasks(rank, world_size, cfg):
    """keep the basic setup list here"""
    setup()
    clear_gpu_cache(rank)  # need to call torch set device first?
    # set_printing()
    setup_environ_flags(cfg, rank)


class ZeroPrint:
    def __init__(self, rank):
        self.rank = rank

    def __call__(self, msg):
        if self.rank == 0:
            print(f"{msg}")


def main():
    seed = 2020
    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # seed with local_rank to ensure we have diff data to all_reduce
    rank_seed = seed + local_rank
    seed_all(rank_seed)

    if rank == 0:
        print(f"--> World Size = {world_size}\n")
        print(f"--> Device_count = {torch.cuda.device_count()}")

    _print = ZeroPrint(rank=local_rank)

    _print(f"all working!!!")

    setup()
    _device = torch.device(f"cuda:{local_rank}")
    _print("setup complete.")

    # -------------- main work --------------
    model = nn.Linear(4096, 4096, bias=False).to(_device)

    compute_tensor = torch.randn((1024, 4096), device=_device)
    comm_tensor = torch.randn((4096, 4096), device=_device)

    compute_iters = 4096
    allreduce_iters = 30

    allreduce_warmup_iters = 10

    comm_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.current_stream()

    torch.cuda.Stream.synchronize(compute_stream)

    allreduce_time_list = []

    # start testing

    # --- all reduce, no overlap

    # Warming up
    prof = None

    _print("Warming up...")
    with torch.cuda.stream(comm_stream):
        for i in range(allreduce_warmup_iters):
            torch.distributed.all_reduce(comm_tensor)
    torch.cuda.Stream.synchronize(comm_stream)
    # prof.step()

    # Profiling allreduce time when not overlapping with computation

    _print("Profiling all_reduce, no overlap ...")
    with torch.cuda.stream(comm_stream):
        for i in range(allreduce_iters):
            torch.distributed.all_reduce(comm_tensor)
    torch.cuda.Stream.synchronize(comm_stream)
    # prof.step()

    # -----------------------------------------

    _print(f"exiting...")
    cleanup()


if __name__ == "__main__":
    main()

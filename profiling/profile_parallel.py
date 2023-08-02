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


from profiling_utils import (
    seed_all,
    RankPrint,
    setup,
    cleanup,
    setup_environ_flags,
    clear_gpu_cache,
)


def setup_tasks(rank, world_size, cfg):
    """keep the basic setup list here"""
    setup()
    clear_gpu_cache(rank)  # need to call torch set device first?
    # set_printing()
    setup_environ_flags(rank)


def main():
    seed = 2020

    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # seed with local_rank to ensure we have diff data to all_reduce
    rank_seed = seed + local_rank
    seed_all(rank_seed)

    setup_tasks(rank=rank, world_size=world_size)

    _print = RankPrint(rank=local_rank, rank_to_print=0)

    cleanup()


if __name__ == "__main__":
    main()

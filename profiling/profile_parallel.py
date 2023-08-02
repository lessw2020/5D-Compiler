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

from config_parallel import ParallelConfig

from profiling_utils import (
    seed_all,
    RankPrint,
    setup,
    cleanup,
    setup_environ_flags,
    clear_gpu_cache,
)


def setup_tasks(
    rank,
    world_size,
):
    """keep the basic setup list here"""
    setup()
    clear_gpu_cache(rank)  # need to call torch set device first?
    # set_printing()
    setup_environ_flags(rank)


class PreModuleSynch(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, hidden_states):
        return hidden_states


class PreMLP(nn.Module):
    def __init__(self, linear_dim: int = 1024):
        super().__init__()
        self.linear = nn.Linear(linear_dim, linear_dim)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        return hidden_states


def group_all_reduce(input, group):
    """All-reduce the the input tensor across model parallel group."""

    if dist.get_world_size(group=group) == 1:
        return input

    # All-reduce.
    dist.all_reduce(input.contiguous(), group=group)

    return input


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

    cfg = ParallelConfig()
    _print(f"{cfg=}")

    cleanup()


if __name__ == "__main__":
    main()

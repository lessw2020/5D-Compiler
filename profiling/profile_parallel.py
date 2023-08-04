# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
# code based on: https://github.com/Hsword/Hetu/tree/main/tools/Galvatron

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
import torch.nn as nn
from config_parallel import ParallelConfig
from profiling_utils import (
    RankPrint,
    cleanup,
    clear_gpu_cache,
    seed_all,
    setup,
    setup_environ_flags,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.pipeline.sync import Pipe
import sys
from tqdm import tqdm

sys.path.append("..")
sys.path.append("../..")

from parallel_utils.comm_utils import gen_groups

import colorama
from colorama import Fore, Style

colorama.init(autoreset=True)


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
    """all_reduce across model parallel group."""

    if dist.get_world_size(group=group) == 1:
        return input

    dist.all_reduce(input.contiguous(), group=group)

    return input


class ReduceModelParallelRegion(torch.autograd.Function):
    """All_reduce the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input, group):
        return group_all_reduce(input, group)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input, group):
        ctx.group = group
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return group_all_reduce(grad_output, ctx.group), None


def reduce_from_tensor_model_parallel_region_group(input, group):
    return ReduceModelParallelRegion.apply(input, group)


def copy_to_tensor_model_parallel_region_group(input, group):
    return CopyToModelParallelRegion.apply(input, group)


class AllReduceBlock(nn.Module):
    def __init__(self, tp_group):
        super().__init__()
        self.tp_group = tp_group
        self.linear = nn.Linear(1024, 1024)

    def forward(self, hidden_states):
        hidden_states = copy_to_tensor_model_parallel_region_group(
            hidden_states, self.tp_group.group
        )
        hidden_states = reduce_from_tensor_model_parallel_region_group(
            hidden_states, self.tp_group.group
        )
        return hidden_states


class DataLoaderRandom(Dataset):
    def __init__(self, cfg):
        self.dataset_size = cfg.bs_local * 8 * 11 // cfg.pp_degree
        # self.input = np.random.randint(0, 100, size=(self.dataset_size, 512, 1024))
        self.input = np.ones((self.dataset_size, 512, 1024))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if idx >= self.dataset_size:
            raise IndexError
        input = torch.FloatTensor(self.input[idx])
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
    device = torch.device("cuda", rank)
    _print(f"{device=}")
    cfg = ParallelConfig()
    _print(f"{cfg=}")

    # * --------------  Training ----------------

    rpc.init_rpc(
        name=f"worker{rank}",
        rank=rank,
        world_size=world_size,
    )

    dataset = DataLoaderRandom(cfg)
    _print(f"{dataset=}")

    trainloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.bs_local,
        sampler=DistributedSampler(dataset, shuffle=False),
    )

    _print(f"len of trainloader {len(trainloader)}")

    all_tp_sizes = [cfg.tp_degree] * 24
    tp_consecutive_flags = [cfg.tp_consecutive] * 24
    tp_groups, _, _, _ = gen_groups(all_tp_sizes, tp_consecutive_flags)
    _print(f"{tp_groups=}")

    model = nn.Sequential()
    model.add_module("pre_sync_module", PreModuleSynch())
    model.add_module("pre_mlp", PreMLP())
    for i in range(len(all_tp_sizes)):
        module = AllReduceBlock(tp_group=tp_groups[i])
        _print(f"adding module mlp_{i}")
        model.add_module(f"mlp_{i}", module)

    avg_num_layers = cfg.model_num_layers // cfg.pp_degree

    pp_degree = cfg.pp_degree

    pp_ranks_enc = []
    for i in range(pp_degree):
        pp_ranks_enc += [i] * avg_num_layers

    devices = [i * world_size + rank for i in range(pp_degree)]
    _print(f"{devices=}")
    pp_devices = [devices[i] for i in pp_ranks_enc]
    model[0] = DDP(model[0].cuda(devices[0]))  # for sync
    model[1] = model[1].cuda(devices[0])
    for i in range(len(all_tp_sizes)):
        model[i + 2] = model[i + 2].cuda(pp_devices[i])

    model = Pipe(model, chunks=1, checkpoint="never")

    optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=0.01)

    # Calculate theoretical communication message size
    tp_size = cfg.tp_degree
    pp_size = cfg.pp_degree
    dp_size = world_size // tp_size
    bs = cfg.bs_local

    allreduce_message_size_per_layer = (
        2 * (tp_size - 1) / tp_size * (bs * 512 * 1024 * 2 * 4 / 1024 / 1024)
    )
    allreduce_message_size_total = allreduce_message_size_per_layer * 24

    _print(f"Strategy: {pp_size}_{tp_size}_{cfg.tp_consecutive}")
    _print(
        f"[allreduce_message_size]: per_layer {allreduce_message_size_per_layer} MB, \ntotal {allreduce_message_size_total} MB"
    )

    def trace_handler(prof):
        if rank == 0:
            table = prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=5
            )
            print(table)
            table = table.split("\n")

            def split_line(line):
                line = line.split("  ")
                ls = []
                for s in line:
                    if len(s):
                        ls.append(s.strip())
                return ls

            def str2time(s):
                if "ms" in s:
                    return float(s[:-2])
                elif "us" in s:
                    return float(s[:-2]) * 1e-3
                else:
                    return float(s[:-1]) * 1e3

            for line in table:
                if "Name" in line:
                    title = split_line(line)
                if "ncclKernel_AllReduce" in line:
                    result = split_line(line)
            for i in range(len(title)):
                # print('%s: %s'%(title[i],result[i]))
                if "CUDA total" in title[i]:
                    cuda_total_idx = i
            allreduce_time_24_layer = str2time(result[cuda_total_idx]) / 10
            comm_coe = allreduce_time_24_layer / allreduce_message_size_total
            inverse_comm_coe = allreduce_message_size_total / allreduce_time_24_layer

            _print(
                f"{Fore.LIGHTBLUE_EX}Strategy:{Style.RESET_ALL} {pp_size}_{tp_size}_{cfg.tp_consecutive}"
            )
            _print(
                f"[allreduce_message_size]: per_layer {allreduce_message_size_per_layer} MB, \ntotal {allreduce_message_size_total} MB"
            )
            print(Fore.LIGHTBLUE_EX + "----  Communication Coefficient ------- ")
            print(
                Fore.LIGHTBLUE_EX
                + f" Parallelism:  {Style.RESET_ALL} Pipeline {Fore.GREEN} {pp_size}{Style.RESET_ALL}, Tensor {Fore.GREEN}{tp_size}{Style.RESET_ALL}, Consecutive {Fore.GREEN} {cfg.tp_consecutive}"
            )
            print(
                f"comm_coe: {pp_size}_{tp_size}_{cfg.tp_consecutive}\n{Fore.LIGHTBLUE_EX}(ms/MB - minimize): {Fore.RED}{comm_coe} {Style.RESET_ALL}\n{Fore.LIGHTBLUE_EX}(MB/ms - maximize):{Fore.GREEN} {inverse_comm_coe}"
            )
            print("----------------------------------------")

            comm_type = "allreduce"
            gpu_num = world_size * pp_size
            env_config_path = "../../env_configs/%s_bandwidth_%d_gpus.json" % (
                comm_type,
                gpu_num,
            )
            config = (
                dict()
            )  # read_json_config(env_config_path) if os.path.exists(env_config_path) else dict()
            key = "%d_%d_%d" % (pp_size, tp_size, cfg.tp_consecutive)
            config[key] = comm_coe
            # write_json_config(config, env_config_path)
            # print('Already written comm_coe_%s into env config file %s!'%(key, env_config_path))

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=10),
        on_trace_ready=trace_handler,
    ) as p:
        for i, input in enumerate(tqdm(trainloader)):
            input = input.to(device)
            out = model(input)
            loss = out.local_value().sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            p.step()

    # * --------------  End Training -------------
    dist.barrier()
    print(f"\nCleaning up and exiting, rank {rank}")
    cleanup()


if __name__ == "__main__":
    main()

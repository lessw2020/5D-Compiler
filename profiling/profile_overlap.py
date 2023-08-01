# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
# code based on: https://github.com/Hsword/Hetu/tree/main/tools/Galvatron

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


_global_allred_time = None
print(f"{id(_global_allred_time)=}")


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
    # if dist.is_initialized():
    #    torch.cuda.set_device(local_rank)

    _device = torch.device(f"cuda:{local_rank}")
    print(f"setup complete, {_device=}.")
    _print(f"inside main {id(_global_allred_time)=}")
    # ----------------------

    def synch_results(
        allreduce_time,
        allreduce_time_list=None,
    ):
        if not allreduce_time:
            _print(f"{rank=}, {allreduce_time=}\n")
            allreduce_time = 0  # 100 + local_rank

        assert allreduce_time, "empty all_reduce_time"
        allreduce_time = torch.tensor([allreduce_time], device=_device)
        dist.reduce(allreduce_time, 0, op=dist.ReduceOp.SUM)

        if local_rank == 0:
            allreduce_time = allreduce_time.cpu().numpy()[0] / world_size
            print("Average allreduce time (ms):", allreduce_time)
            allreduce_time_list.append(allreduce_time)
        print(f"End reduce synch")

    # trace handler
    def trace_handler(
        prof,
    ):
        global _global_allred_time

        table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=5)
        if rank == 3 or rank == 0:
            print(f"\n{rank=}, {table=}\n")
        table = table.split("\n")
        # _print(f"{table=}")
        result = []

        for line in table:
            # if rank == 0 or rank == 3:
            #    print(f"{line=}")
            if "Name" in line:
                title = split_line(line)
            if "ncclKernel_AllReduce" in line:
                print(f"result found on {rank=}")
                result = split_line(line)
                # _print(f"{result=}")

        for i in range(len(title)):
            if "CUDA total" in title[i]:
                cuda_total_idx = i
            elif "# of Calls" in title[i]:
                call_times_idx = i

        if not result:
            print(f"{rank=}, ****  empty result in trace handler")
            return
        assert result, f"empty result"
        _print(f"{title=}")
        _print(f"{result=}, \n {len(result)}, {cuda_total_idx=}, {call_times_idx=}\n\n")
        cuda_totals = result[cuda_total_idx]
        call_times_totals = result[call_times_idx]
        _print(f"\n{cuda_totals=}, {call_times_totals=}\n")

        allreduce_time = str2time(result[cuda_total_idx]) / int(result[call_times_idx])
        final_allred_time = round(allreduce_time, 8)
        print(f"{rank=}, {final_allred_time=}")
        _global_allred_time = final_allred_time

        dist.barrier()
        synch_results(final_allred_time, allreduce_time_list)

    # -------------- main work --------------
    model = nn.Linear(4096, 4096, bias=False).to(_device)

    compute_tensor = torch.randn((1024, 4096), device=_device)
    comm_tensor = torch.randn((4096, 4096), device=_device)
    comm_tensor2 = torch.randn((4096, 4096), device=_device)

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

    def warmup_allreduce(comm_stream, comm_tensor, num_warmups):
        print(f"Warming up...{rank=}")
        with torch.cuda.stream(comm_stream):
            for i in range(num_warmups):
                dist.all_reduce(comm_tensor, async_op=True)
        torch.cuda.Stream.synchronize(comm_stream)
        _print(f"Warmup completed")

    # prof.step()

    # Profiling allreduce time when not overlapping with computation

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=1),
        on_trace_ready=trace_handler,
    ) as prof:
        _print("Profiling all_reduce, no overlap ...")
        warmup_allreduce(comm_stream, comm_tensor, num_warmups=allreduce_iters)
        dist.barrier()
        prof.step()
        _print(f"pure all_reduce, warmup profiling done")

        with torch.cuda.stream(comm_stream):
            for i in range(allreduce_iters):
                dist.all_reduce(comm_tensor, async_op=False)
        torch.cuda.Stream.synchronize(comm_stream)
        prof.step()
        # print(f"about to synch pure: {rank=}, {_global_allred_time=}")
        # synch_results(_global_allred_time, allreduce_time_list)

    _print(f"Success - profiled all_reduce pure mode")
    dist.barrier()
    _print(f"{id(_global_allred_time)=}")
    print(f"{rank=}, after pure allred {_global_allred_time=}")

    # profile overlaps...
    _print("Profiling overlapping...")

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=1),
        on_trace_ready=trace_handler,
    ) as prof:
        warmup_allreduce(comm_stream, comm_tensor, num_warmups=allreduce_iters)
        dist.barrier()
        prof.step()
        _print(f"past overlap warmup...")

        with torch.cuda.stream(comm_stream):
            for i in range(allreduce_iters):
                dist.all_reduce(comm_tensor, async_op=True)

        with torch.cuda.stream(compute_stream):
            for i in range(compute_iters):
                output = model(compute_tensor)

        dist.barrier()
        torch.cuda.Stream.synchronize(comm_stream)
        prof.step()
    dist.barrier()
    # synch_results(_global_allred_time, allreduce_time_list)

    # -----------------------------------------

    _print(f"exiting...")
    cleanup()


if __name__ == "__main__":
    main()

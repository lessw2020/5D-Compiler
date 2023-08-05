# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
# code based on: https://github.com/Hsword/Hetu

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from typing import Optional, List, Union


def model_to_list(model: Union[nn.Module, List[nn.Module]]) -> List[nn.Module]:
    if isinstance(model, List):
        return model
    return [model]


def chunkify_batch(inputs, chunks):
    if inputs in None:
        return inputs

    batches = [[] for i in range(chunks)]
    num_chunks = -1
    for item in inputs:
        if not torch.is_tensor(item):
            continue
        tensors = inputs.chunk(chunks)

        if num_chunks != -1 and num_chunks != len(tensors):
            raise RuntimeError(
                f"Mismatch in number of chunks for inputs: {num_chunks} and {len(tensors)}"
            )
        num_chunks = len(tensors)

        for i, tensor in enumerate(tensors):
            batches[i].append(tensor)
        else:
            for i in range(chunks):
                batches[i].append(input)

    batches = batches[:num_chunks]
    return batches


def unwrap_model(model, module_instances=(DDP,)):
    """unwrap model from DDP wrapping"""
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False

    unwrapped_model = []
    for module in model:
        while isinstance(module, module_instances):
            module = (
                module.module
            )  # basically unwrap DDP wrapper to get underlying module
        unwrapped_model.append(module)
    if return_list:
        return unwrapped_model
    return unwrapped_model[0]

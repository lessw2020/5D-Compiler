# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
# code based on: https://github.com/Hsword/Hetu

import sys
import torch
import torch.nn as nn
import operator
import copy
from torch import Tensor
from typing import Union, Optional, Tuple, List
import torch.distributed as dist
from parallel_utils.pipeline_utils import unwrap_model, chunkify_batch
import operator
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import wrap_modules_data_parallel
from torch.distributed.fsdp import fully_sharded_data_parallel as FSDP

from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.data_parallel import TrainingState

sys.append("..")

Shape = Union[Tuple[int], List[int], torch.Size]


class PipelineParallel(nn.Module):
    def __init__(
        self,
        model,
        model_ranks,
        layer_output_tensor_shapes,
        layer_output_tensor_dtypes=None,
        chunks=1,
        process_group=None,
        num_proc_per_node=None,
        info=False,
    ):
        super().__init__()
        self.total_model_len = len(model)
        assert len(model) == len(model_ranks)
        assert len(model) == len(layer_output_tensor_shapes)

        layer_output_tensor_dtypes = (
            self.get_default_tensor_dtype(layer_output_tensor_shapes)
            if layer_output_tensor_dtypes is None
            else layer_output_tensor_dtypes
        )
        self.check_tensor_dtype(layer_output_tensor_shapes, layer_output_tensor_dtypes)
        if layer_dp_sizes is None:
            layer_dp_sizes = [1] * len(model)
        assert len(model) == len(layer_dp_sizes)
        self.world_size = dist.get_world_size
        self.global_rank = dist.get_rank()
        device_count = torch.cuda.device_count()
        self.device_count = (
            num_proc_per_node
            if num_proc_per_node and num_proc_per_node <= device_count
            else device_count
        )
        self.local_rank = self.global_rank % self.device_count

        self.pp_global_ranks = (
            [i for i in range(self.world_size)]
            if process_group is None
            else sorted(list(set(list(process_group))))
        )

        assert (
            self.global_rank in self.pp_global_ranks
        ), f"global rank {self.global_rank} not in pp_global_ranks {self.pp_global_ranks}"

        self.group = dist.new_group(process_group)
        self.group_size = dist.get_world_size(self.group)
        self.group_rank = dist.get_rank(self.group)

        assert (
            len(list(set(model_ranks))) == self.group_size
            and torch.max(model_ranks) == self.group_size - 1
            and torch.min(model_ranks) == 0
        )

        self.stage_start_idx, count = model_ranks.index(
            self.group_rank
        ), model_ranks.count(self.group_rank)

        self.stage_end_index = self.stage_start_idx + count
        self.model_cur_stage = model[self.stage_start_idx : self.stage_end_index].cuda(
            self.local_rank
        )
        self.chunks = int(chunks)
        assert self.chunks >= 1

        _stage_index = self.stage_start_idx - 1
        _is_first_stage = self.is_pipeline_first_stage()
        _is_last_stage = self.is_pipeline_last_stage()

        self.stage_input_tensor_shape = (
            [None] if _is_first_stage else layer_output_tensor_shapes[_stage_index]
        )

        self.stage_output_tensor_dtype = (
            [None] if _is_first_stage else layer_output_tensor_dtypes[_stage_index]
        )

        self.stage_input_tensor_dtype = (
            [None] if _is_first_stage else layer_output_tensor_dtypes[_stage_index]
        )

        self.stage_output_tensor_dtype = (
            [None]
            if _is_last_stage
            else layer_output_tensor_dtypes[self.stage_end_index - 1]
        )

        self.dp_size_prev_stage = (
            None if _is_first_stage else layer_dp_sizes[_stage_index]
        )

        self.dp_size_cur_stage = (
            None if _is_last_stage else layer_dp_sizes[self.stage_end_index - 1]
        )

        self.dp_size_input = layer_dp_sizes[0]
        self.info = info
        self.chunk_warning = True

    def check_tensor_dtype(
        self, layer_output_tensor_shapes, layer_output_tensor_dtypes
    ):
        assert len(layer_output_tensor_shapes) == len(layer_output_tensor_dtypes)

        for i in range(len(layer_output_tensor_shapes)):
            if layer_output_tensor_shapes[i] is not None:
                assert len(
                    layer_output_tensor_shapes[i] == len(layer_output_tensor_dtypes[i])
                )

    def get_default_tensor_dtype(self, layer_output_tensor_shapes):
        layer_output_tensor_dtypes = []
        for tensor_shape in layer_output_tensor_shapes:
            if tensor_shape is None:
                layer_output_tensor_dtypes.append(None)
            else:
                layer_output_tensor_dtypes.append([torch.float] * len(tensor_shape))
        return layer_output_tensor_dtypes

    def wrap_pipeline_modules_data_parallel(self, dp_types, dp_groups, module_types):
        assert self.total_model_len == len(dp_types)
        assert self.total_model_len == len(dp_groups)
        assert self.total_model_len == len(module_types)

        dp_types_cur_stage = dp_types[self.stage_start_idx : self.stage_end_index]
        module_types_cur_stage = dp_types[self.stage_start_idx : self.stage_end_index]
        dp_groups_cur_stage = dp_groups[self.stage_start_idx : self.stage_end_index]
        pp_devices_cur_stage = [self.local_rank] * (
            self.stage_end_index - self.stage_start_idx
        )
        self.model_cur_stage = wrap_modules_data_parallel(
            self.model_cur_stage,
            dp_types_cur_stage,
            dp_groups_cur_stage,
            module_types=module_types_cur_stage,
            pp_devices=pp_devices_cur_stage,
        )

    def update_tensor_shape(
        self, microbatches, dp_size_input, dp_size, template_tensor_shape
    ):
        """update tensor_shape with correct microbatch size"""
        dp_chunk = dp_size_input // dp_size

        tensor_shape, tensor_shape_last = copy.deepcopy(
            template_tensor_shape
        ), copy.deepcopy(template_tensor_shape)
        microbatch_size = microbatches[0][0][0].shape[0] * dp_chunk
        microbatch_size_last = microbatches[0][-1][0].shape[0] * dp_chunk

        for i in range(len(tensor_shape)):
            tensor_shape[i][0] = microbatch_size
            tensor_shape_last[i][0] = microbatch_size_last
        return tensor_shape, tensor_shape_last

    def gpipe_foward_backward(
        self,
        forward_step_func,
        batch,
        forward_only=False,
    ):
        """Runs gpipe schedule, with comms between pipeline stages.
        Returns: dict with losses if last stage
        """
        losses_reduced = self.gpipe_forward(forward_step_func, batch, forward_only)
        if not forward_only:
            self.gpipe_backward()
        return losses_reduced

    def gpipe_forward(self, forward_step_func, batch, forward_only=False):
        model = self.model_cur_stage
        # create microbatches
        microbatches = [
            chunkify_batch(batch[0], self.chunks),
            chunkify_batch(batch[1], self.chunks),
        ]
        self.real_chunks = len(microbatches[0])
        if self.chunks != self.real_chunks and self.chunk_warning:
            if self.global_rank == 0:
                print(
                    f"\nWarning from PipelineParallel Module: Real chunks is {self.real_chunks}, microbatch sizes are {[m[0].shape[0] for m in microbatches[0]]}"
                )
            self.chunk_warning = False
        self.num_microbatches = self.real_chunks

        # compute tensor shapes for all microbatches. last microbatch may have different size
        batch_size = batch[0][0].shape[0] * self.dp_size_input

        if self.is_pipeline_first_stage():
            self.stage_input_tensor_shape = self.stage_input_tensor_shape_last = [None]
        else:
            (
                self.stage_input_tensor_shape,
                self.stage_input_tensor_shape_last,
            ) = self.update_tensor_shape(
                microbatches,
                self.dp_size_input,
                self.dp_size_prev_stage,
                self.stage_input_tensor_shape,
            )
        if self.is_pipeline_last_stage():
            self.stage_output_tensor_shape = self.stage_output_tensor_shape_last = [
                None
            ]
        else:
            (
                self.stage_output_tensor_shape,
                self.stage_output_tensor_shape_last,
            ) = self.update_tensor_shape(
                microbatches,
                self.dp_size_input,
                self.dp_size_cur_stage,
                self.stage_output_tensor_shape,
            )
        self.input_tensors = []
        self.output_tensors = []
        losses_reduced = []

        if self.info:
            print(f"{self.global_rank}, starting pipeline forward")

        # Forward passes
        for i in range(self.num_microbatches):
            if i == self.num_microbatches - 1:
                recv_tensor_shapes = self.stage_input_tensor_shape_last
                send_tensor_shapes = self.stage_output_tensor_shape_last
            else:
                recv_tensor_shapes = self.stage_input_tensor_shape
                send_tensor_shapes = self.stage_output_tensor_shape
            recv_tensor_dtypes = self.stage_input_tensor_dtype
            send_tensor_dtypes = self.stage_output_tensor_dtype

            input_tensor = self.recv_forward_multi(
                tensor_shapes=recv_tensor_shapes, dtypes=recv_tensor_dtypes
            )
            cur_microbatch = [microbatches[0][i], microbatches[1][i]]

            if self.num_microbatches > 1:
                for m in self.model_cur_stage.modules():
                    # For DDP module, disable grad sync for accumulation and manually sync before
                    # backward of last microbatch
                    if isinstance(m, DDP) and i == 0:
                        m.require_backward_grad_sync = False

            output_tensor = self.forward_step(
                forward_step_func, cur_microbatch, model, input_tensor, losses_reduced
            )
            self.send_forward_multi(
                output_tensor,
                tensor_shapes=send_tensor_shapes,
                dtypes=send_tensor_dtypes,
            )

            if not forward_only:
                self.input_tensors.append(input_tensor)
                self.output_tensors.append(output_tensor)

        if self.info:
            print(f"{self.global_rank} finished forward")
        return losses_reduced

    def gpipe_backward(self):
        if self.info:
            print(f"{self.global_rank} starting backwards")

        for i in range(self.num_microbatches):
            input_tensor = self.input_tensors.pop(0)
            output_tensor = self.output_tensors.pop(0)

            if i == self.num_microbatches - 1:
                recv_tensor_shapes = self.stage_input_tensor_shape_last
                send_tensor_shapes = self.stage_output_tensor_shape_last
            else:
                recv_tensor_shapes = self.stage_input_tensor_shape
                send_tensor_shapes = self.stage_output_tensor_shape

            recv_tensor_dtypes = self.stage_input_tensor_dtype
            send_tensor_dtypes = self.stage_output_tensor_dtype
            output_tensor_grad = self.recv_backward_multi(
                tensor_shapes=send_tensor_shapes, dtypes=send_tensor_dtypes
            )

            if self.num_microbatches > 1:
                for m in self.model_cur_stage.modules():
                    if isinstance(m, DDP) and i == self.num_microbatches - 1:
                        m.require_forward_param_sync = True
                        m.reducer.prepare_for_backward([])

                    # for FSDP, need to disable post backward hooks for accumulation
                    # and register manually before backward of last microbatch
                    elif isinstance(m, FSDP):
                        if i == self.num_microbatches - 1:
                            m.training_state = TrainingState.IDLE
                            m._post_backward_callback_queued = False
                            m._register_post_backward_hooks()
                        else:
                            for p in m.params:
                                if not p.requires_grad:
                                    continue
                                if hasattr(p, "_shard_bwd_hook"):
                                    p._shard_bwd_hook[1].remove()
                                    delattr(p, "_shard_bwd_hook")
                            m._post_bacward_callback_queued = True
            input_tensor_grad = self.backward_step(
                input_tensor, output_tensor, output_tensor_grad
            )
            self.send_backward_multi(
                input_tensor_grad,
                tensor_shapes=recv_tensor_shapes,
                dtypes=recv_tensor_dtypes,
            )

        if self.info:
            print(f"{self.global_rank} finished backwards")

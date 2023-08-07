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

    def listify_(self, input_tensor):
        if not isinstance(input_tensor, (list, tuple)):
            input_tensor = [input_tensor]

    def forward_step(
        self,
        forward_step_func,
        batch,
        model,
        input_tensor,
        losses_reduced,
        loss_stage=False,
    ):
        """input tensor is from data_iterator for first stage, otherwise, input tensor passed in
        returns output_tensor
        """
        unwrapped_model = unwrap_model(model)
        self.listify_(input_tensor)

        for x in input_tensor:
            if x is not None and x.dtype == torch.float32:
                x.requires_grad = True

        if input_tensor[0] is None:
            output_tensor, loss_func = forward_step_func(batch[0], model)
        else:
            output_tensor, loss_func = forward_step_func(input_tensor, model)

        if self.is_pipeline_last_stage():
            self.listify_(output_tensor)
            output_tensor = loss_func(batch[1], output_tensor)
            loss = output_tensor
            output_tensor = loss / self.real_chunks
            losses_reduced.append(loss)
            return output_tensor

    def backward_step(self, input_tensor, output_tensor, output_tensor_grad):
        """backwards via output_tensor
        Returns gradient of loss with respect to input tensor (None if first
        stage)
        """

        # self.listify_(input_tensor)
        unwrap_input_tensor_grad = not isinstance(input_tensor, list)
        if unwrap_input_tensor_grad:
            input_tensor = [input_tensor]
        i
        input_tensor = [
            None if t is None or not t.requires_grad else t for t in input_tensor
        ]
        for x in input_tensor:
            if x is not None:
                x.retain_grad()

        self.listify_(output_tensor)
        self.listify_(output_tensor_grad)

        # backward pass
        output_tensor_, output_tensor_grad_ = [], []
        for t, g in zip(output_tensor, output_tensor_grad):
            if t is not None and t.requires_grad():
                output_tensor_.append(t)
                output_tensor_grad_.append(g)
        torch.autograd.backward(output_tensor_, grad_tensors=output_tensor_grad_)

        # collect grads
        input_tensor_grad = [None]
        if input_tensor is not None:
            input_tensor_grad = []
            for x in input_tensor:
                input_tensor_grad.append(None if x is None else x.grad)

        return input_tensor_grad[0] if unwrap_input_tensor_grad else input_tensor_grad

    # ------- pipeline rank utils --------------------
    def get_pipeline_mp_first_rank(
        self,
    ):
        return self.pp_global_ranks[0]

    def get_pipeline_mp_last_rank(
        self,
    ):
        last_rank_local = self.group_size - 1
        return self.pp_global_ranks[last_rank_local]

    def get_pipeline_mp_next_rank(
        self,
    ):
        rank_in_pipeline = self.group_rank
        world_size = self.group_size
        return self.pp_global_ranks[(rank_in_pipeline + 1) % world_size]

    def get_pipeline_mp_prev_rank(
        self,
    ):
        rank_in_pipeline = self.group_rank
        world_size = self.group_size
        return self.pp_global_ranks[(rank_in_pipeline - 1) % world_size]

    def is_pipeline_first_stage(self) -> bool:
        return self.group_rank == 0

    # ------ pp comm utils ------------------
    def run_p2pops(
        self,
        tensor_send_prev: Union[torch.Tensor, None],
        tensor_send_next: Union[torch.Tensor, None],
        tensor_recv_prev: Union[torch.Tensor, None],
        tensor_recv_next: Union[torch.Tensor, None],
    ):
        ops = []
        if tensor_send_prev is not None:
            send_prev_op = dist.P2POp(
                dist.isend,
                tensor_send_prev,
                self.get_pipeline_mp_prev_rank(),
            )
            ops.append(send_prev_op)

        if tensor_recv_prev is not None:
            recv_prev_op = dist.P2POp(
                dist.irecv,
                tensor_recv_prev,
                self.get_pipeline_mp_prev_rank(),
            )
            ops.append(recv_prev_op)

        if tensor_send_next is not None:
            send_next_op = dist.P2POp(
                dist.isend, tensor_send_next, self.get_pipeline_mp_next_rank()
            )
            ops.append(send_next_op)

        if tensor_recv_next is not None:
            recv_next_op = dist.P2POp(
                dist.irecv, tensor_recv_next, self.get_pipeline_mp_next_rank()
            )
            ops.append(recv_next_op)

        if len(ops):
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

    def communicate(
        self,
        tensor_send_next: Optional[torch.Tensor],
        tensor_send_prev: Optional[torch.Tensor],
        recv_prev: bool,
        recv_next: bool,
        tensor_shape: Optional[Shape] = None,
        override_scatter_gather_tensors_in_pipeline: bool = False,
        dtype_: Optional[torch.dtype] = None,
        *,
        scatter_gather_tensors_in_pipeline: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        fp32_residual_connection: bool = False,
    ) -> Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        """Main function for comm of tensor between stages"""
        tensor_recv_prev = None
        tensor_recv_next = None
        if tensor_shape is None:
            raise RuntimeError(
                "tensor shape must be speced. Usually (seq_length, microbatch_size, hidden_size)"
            )
        if (
            not override_scatter_gather_tensors_in_pipeline
            and scatter_gather_tensors_in_pipeline
        ):
            assert False, f"not implemented yet"
            # tensor_chunk_shape = (
            #    reduce(operator.mul, tensor_shape, 1)
            #    // self.get_tensor_model_parallel_world_size(),
            # )
        else:
            tensor_chunk_shape = tensor_shape

        # from Megatron-LM
        dtype = params_dtype or torch.float
        if fp32_residual_connection:
            dtype = torch.float
        requires_grad = True
        if dtype_ is not None:
            dtype = dtype_
            requires_grad = False

        if recv_prev:
            tensor_recv_prev = torch.empty(
                tensor_chunk_shape,
                requires_grad=requires_grad,
                device=torch.cuda.current_device(),
                dtype=dtype,
            )

        if recv_next:
            tensor_recv_next = torch.empty(
                tensor_chunk_shape,
                requires_grad=requires_grad,
                device=torch.cuda.current_device(),
                dtype=dtype,
            )

        if (
            not override_scatter_gather_tensors_in_pipeline
            and scatter_gather_tensors_in_pipeline
        ):
            if tensor_send_next is not None:
                tensor_send_next = split_tensor_into_1d_equal_chunks(tensor_send_next)

            if tensor_send_prev is not None:
                tensor_send_prev = split_tensor_into_1d_equal_chunks(tensor_send_prev)

        commtype = p2p_type(
            tensor_send_prev, tensor_send_next, tensor_recv_prev, tensor_recv_next
        )

        if self.info:
            print(f"{self.global_rank} starting p2p with {commtype=}")

        # send tensors forward and backward
        self.run_p2pops(
            tensor_send_prev, tensor_send_next, tensor_recv_prev, tensor_recv_next
        )
        # protect from race when using batch_isend_irecv:
        torch.cuda.synchronize()

        if self.info:
            print(f"{self.global_rank} finished p2p, {commtype=}")

        if (
            not override_scatter_gather_tensors_in_pipeline
            and scatter_gather_tensors_in_pipeline
        ):
            if recv_prev:
                tensor_recv_prev = (
                    gather_split_1d_tensor(tensor_recv_prev)
                    .view(tensor_shape)
                    .requires_grad_()
                )
            if recv_next:
                tensor_recv_next = (
                    gather_split_1d_tensor(tensor_recv_next)
                    .view(tensor_shape)
                    .requires_grad_()
                )

        return tensor_recv_prev, tensor_recv_next

    def recv_forward(
        self,
        tensor_shape: Shape,
        override_scatter_gather_tensors_in_pipeline: bool = False,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Receive tensor from previous rank in pipeline (fwd recv)"""
        if self.is_pipeline_first_stage():
            return None
        input_tensor, _ = self.communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shape,
            override_scatter_gather_tensors_in_pipeline=override_scatter_gather_tensors_in_pipeline,
            dtype_=dtype,
        )
        return input_tensor

    def recv_backward(
        self,
        tensor_shape: Shape = None,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """recv tensor from next rank in pipeline (backward receive)"""
        if self.is_pipeline_last_stage():
            return None
        _, output_tensor_grad = self.communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shape=tensor_shape,
            dtype_=dtype,
        )
        return output_tensor_grad

    def send_forward(
        self,
        output_tensor: torch.Tensor,
        override_scatter_gather_tensors_in_pipeline: bool = False,
        tensor_shape: Shape = None,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Send tensor to next rank in pipeline (forward)"""
        if self.is_is_pipeline_last_stage():
            return
        self.communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=False,
            override_scatter_gather_tensors_in_pipeline=override_scatter_gather_tensors_in_pipeline,
            tensor_shape=tensor_shape,
            dtype_=dtype,
        )

    def send_backward(
        self,
        input_tensor_grad: torch.Tensor,
        tensor_shape: Shape,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Send tensor backwrd to previous rank"""
        if self.is_pipeline_first_stage():
            return
        self.communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=False,
            recv_next=False,
            tensor_shape=tensor_shape,
            dtype_=dtype,
        )

    def send_forward_recv_backward(
        self,
        output_tensor: torch.Tensor,
        tensor_shape: Shape,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> Union[None, torch.Tensor]:
        """two way forward with next rank - send forward, receive backward"""
        if self.is_pipeline_last_stage():
            return None
        output_tensor_grad = self.communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shape=tensor_shape,
            dtype_=dtype,
        )
        return output_tensor_grad

    def send_backward_recv_forward(
        self,
        input_tensor_grad: torch.Tensor,
        tensor_shape: Shape,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> Union[None, torch.Tensor]:
        """two way backward with prev rank - send backward, receive forward"""
        if self.is_pipeline_last_stage():
            return None
        input_tensor, _ = self.communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shape,
            dtype_=dtype,
        )
        return input_tensor
    
    def send_forward_recv_forward(
            self, 
            output_tensor: torch.Tensor,
            recv_prev: bool,
            tensor_shape: Shape,
            *,
            dtype: Optional[torch.dtype]=None,
    )
        """batched recv from prev, forward to next rank"""
        input_tensor,_ = self.communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev = recv_prev,
            recv_next = False,
            tensor_shape = tensor_shape,
            dtype_=dtype
        )
        return input_tensor
    
    def send_backward_recv_backward(self,
                                    input_tensor_grad: torch.Tensor,
                                    recv_next: bool,
                                    tensor_shape: Shape,
                                    *,
                                    dtype: Optional[torch.dtype]=None,)->torch.Tensor:
        """batch recv backward from next rank, send to prev rank"""
        _, output_tensor_grad = self.communicate(
            tensor_send_next=None,
            tensor_send_prev = input_tensor_grad,
            recv_prev=False,
            recv_next=recv_next,
            tensor_shape = tensor_shape,
            dtype_=dtype,

        )
        return output_tensor_grad
    
    def send_forward_backward_recv_forward_backward(
            self,
            output_tensor: torch.Tensor,
            input_tensor_grad: torch.Tensor,
            recv_prev: bool,
            recv_next: bool,
            tensor_shape: Shape,
            *,
            dtype: Optional[torch.dtype]=None,)->Tuple[torch.Tensor, torch.Tensor]:
        """Batched send and recv with both prev and next ranks"""
        input_tensor, output_tensor_grad = self.communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=input_tensor_grad,
            recv_prev=recv_prev,
            recv_next=recv_next,
            tensor_shape = tensor_shape,
            dtype_=dtype,

        )
        return input_tensor, output_tensor_grad
    
    """ 
    

        
        
    """


class PipeSequential(nn.Sequential):
    """pipeline variant of nn.sequential"""

    def forward(self, *inputs):
        for module in self:
            if isinstance(inputs, Tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

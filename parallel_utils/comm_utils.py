# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
# code based on: https://github.com/Hsword/Hetu

import torch.distributed as dist


def show_groups(groups):
    for group in groups:
        if group is None:
            print("None", end=" ")
        else:
            group.print()
    print()


class CommGroup(object):
    def __init__(self, ranks):
        assert isinstance(
            ranks, (list, range)
        ), f"List or range of ranks should be provided here to create CommGroup, received {type(ranks)}!"

        self.ranks = sorted(list(set(list(ranks))))
        self.size = len(self.ranks)
        self.group = dist.new_group(self.ranks)

    def has_rank(self, rank):
        if rank in self.ranks:
            self.intra_group_id = self.ranks.index(rank)
            return True
        return False

    def allgather(self, input):
        return gather_from_tensor_model_parallel_region_group(input, self.group)

    def print(self):
        print(self.ranks, end=" ")


class SliceFunc(object):
    def __init__(self, slice_num, local_rank):
        self.n = slice_num
        self.local_rank = local_rank
        assert local_rank < slice_num

    def __call__(self, input):
        length = len(input)
        step = int(length // self.n)
        return input[int(self.local_rank * step) : int((self.local_rank + 1) * step)]

    def print(self):
        print(f"{self.local_rank}/{self.n}", end=" ")


def gen_groups(all_tp_sizes, tp_consecutive_flags, show_rank=-1):
    world_size = dist.get_world_size()
    for i in range(len(all_tp_sizes)):
        tp_consec = tp_consecutive_flags[i]
        assert tp_consec == 0 or tp_consec == 1
        if all_tp_sizes[i] in [1, world_size]:
            tp_consecutive_flags[i] = 1
    tp_groups = []
    dp_groups = []
    allgather_groups = [None]
    slice_funcs = [None]
    tp_group_dict_consec = get_tp_group_dict(all_tp_sizes, True)
    tp_group_dict_inconsec = get_tp_group_dict(all_tp_sizes, False)
    dp_group_dict_consec = get_dp_group_dict(all_tp_sizes, True)
    dp_group_dict_inconsec = get_dp_group_dict(all_tp_sizes, False)

    for i in range(len(all_tp_sizes)):
        if tp_consecutive_flags[i]:
            tp_groups.append(tp_group_dict_consec[all_tp_sizes[i]])
            dp_groups.append(dp_group_dict_inconsec[all_tp_sizes[i]])
        else:
            tp_groups.append(tp_group_dict_inconsec[all_tp_sizes[i]])
            dp_groups.append(dp_group_dict_consec[all_tp_sizes[i]])

    for i in range(1, len(all_tp_sizes)):
        allgather_groups.append(
            gen_allgather_group(
                all_tp_sizes[i - 1],
                all_tp_sizes[i],
                tp_consecutive_flags[i - 1],
                tp_consecutive_flags[i],
                tp_groups[i],
            )
        )
    for i in range(1, len(all_tp_sizes)):
        slice_funcs.append(
            gen_slice_func(
                all_tp_sizes[i - 1],
                all_tp_sizes[i],
                tp_consecutive_flags[i - 1],
                tp_consecutive_flags[i],
                tp_groups[i - 1],
            )
        )
    if show_rank >= 0 and dist.get_rank() == show_rank:
        print("-------- 5D Communication Group -------")
        print(f"TP groups for rank {show_rank} (all layers):")
        show_groups(tp_groups)
        print("DP groups for rank {show_rank} (all layers):")
        show_groups(dp_groups)

        # print("AllGather groups for rank %d:"%show_rank)
        # show_groups(allgather_groups)
        # print("Slice Funcs for rank %d:"%show_rank)
        # show_groups(slice_funcs)
        print("---------------------------------------")
    return tp_groups, dp_groups, allgather_groups, slice_funcs

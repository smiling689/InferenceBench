#!/usr/bin/env python3
import argparse
import socket
import time
from types import SimpleNamespace

import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing

from sglang.srt.managers.prefill_delayer import PrefillDelayer, _State


def check_rank(rank, world_size, port, patched):
    distributed.init_process_group(
        'gloo', init_method=f'tcp://127.0.0.1:{port}',
        rank=rank, world_size=world_size,
    )
    server_args = SimpleNamespace(
        prefill_delayer_queue_min_ratio=0.15,
        prefill_delayer_max_delay_ms=350.0,
        enable_dp_attention=False,
        disable_overlap_schedule=False,
    )
    delayer = PrefillDelayer(
        dp_size=1, attn_tp_size=world_size,
        cpu_group=distributed.group.WORLD, server_args=server_args,
        max_delay_passes=30, token_usage_low_watermark=None,
    )
    delayer.skip_first_delayer = False
    outcomes = []
    for leader_expired in [True, False]:
        expired = leader_expired if rank == 0 else not leader_expired
        state = _State(start_time=time.perf_counter() - (1.0 if expired else 0.0))
        output = delayer._negotiate_should_allow_prefill_pure(
            prev_state=state, local_prefillable=True, token_usage=0.1,
            running_batch=15, max_prefill_bs=4,
            max_running_requests=128, waiting_queue_len=1,
        )
        local = torch.tensor([int(output.output_allow)], dtype=torch.int64)
        gathered = [torch.empty_like(local) for _ in range(world_size)]
        distributed.all_gather(gathered, local)
        decisions = [int(value.item()) for value in gathered]
        if patched:
            assert decisions == [int(leader_expired)] * world_size, decisions
        else:
            assert len(set(decisions)) == 2, decisions
        assert delayer._max_delay_ms == 350.0
        outcomes.append(decisions)
    if rank == 0:
        print({'patched': patched, 'world_size': world_size, 'decisions': outcomes}, flush=True)
    distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patched', action='store_true')
    args = parser.parse_args()
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    multiprocessing.spawn(check_rank, args=(4, port, args.patched), nprocs=4, join=True)


if __name__ == '__main__':
    main()

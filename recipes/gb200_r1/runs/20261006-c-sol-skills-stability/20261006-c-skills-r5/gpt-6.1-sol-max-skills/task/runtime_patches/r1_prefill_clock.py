import time

import torch
import torch.distributed as distributed

from sglang.srt.managers.prefill_delayer import PrefillDelayer


def install():
    original = PrefillDelayer._negotiate_should_allow_prefill_pure
    if getattr(original, '_r1_coherent_clock', False):
        return

    def synchronized_timeout(self, prev_state, *args, **kwargs):
        if not self._queue_trigger_enabled or prev_state is None:
            return original(self, prev_state, *args, **kwargs)
        if distributed.get_world_size(self._gather_group) == 1:
            return original(self, prev_state, *args, **kwargs)
        timeout_reached = (
            (time.perf_counter() - prev_state.start_time) * 1000.0
            >= self._max_delay_ms
        )
        if not hasattr(self, '_r1_clock_buffer'):
            self._r1_clock_buffer = torch.empty(
                1, dtype=torch.int64, device=self._gather_device
            )
            self._r1_clock_leader = distributed.get_process_group_ranks(
                self._gather_group
            )[0]
        self._r1_clock_buffer.fill_(int(timeout_reached))
        distributed.broadcast(
            self._r1_clock_buffer,
            src=self._r1_clock_leader,
            group=self._gather_group,
        )
        original_limit = self._max_delay_ms
        self._max_delay_ms = (
            0.0 if self._r1_clock_buffer.item() else float('inf')
        )
        try:
            return original(self, prev_state, *args, **kwargs)
        finally:
            self._max_delay_ms = original_limit

    synchronized_timeout._r1_coherent_clock = True
    PrefillDelayer._negotiate_should_allow_prefill_pure = synchronized_timeout

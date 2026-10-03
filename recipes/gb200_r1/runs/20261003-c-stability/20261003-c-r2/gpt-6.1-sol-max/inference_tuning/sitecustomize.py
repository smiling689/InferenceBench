"""Deterministic TP-local admission batching; model execution is unchanged."""

import importlib.abc
import importlib.machinery
import logging
import os
import sys


def patch_scheduler(module):
    scheduler_class = module.Scheduler
    original_method = scheduler_class._get_new_batch_prefill_raw
    max_wait_steps = max(1, int(os.environ.get("R1_PREFILL_BATCH_MAX_WAIT_STEPS", "80")))
    target_divisor = max(1, int(os.environ.get("R1_PREFILL_BATCH_TARGET_DIVISOR", "4")))

    def batched_prefill(self, prefill_delayer_single_pass=None):
        if self.server_args.enable_dp_attention or self.server_args.dp_size != 1:
            return original_method(self, prefill_delayer_single_pass)

        running_requests = len(self.running_batch.reqs)
        queued_requests = len(self.waiting_queue)
        if self.chunked_req is None and running_requests > 0 and queued_requests > 0:
            target_batch = min(8, max(2, running_requests // target_divisor))
            wait_steps = getattr(self, "_local_prefill_wait_steps", 0)
            if queued_requests < target_batch and wait_steps < max_wait_steps:
                self._local_prefill_wait_steps = wait_steps + 1
                return None

        self._local_prefill_wait_steps = 0
        return original_method(self, prefill_delayer_single_pass)

    scheduler_class._get_new_batch_prefill_raw = batched_prefill
    logging.getLogger(__name__).warning(
        "Enabled deterministic TP-local prefill batching (max_wait_steps=%s, divisor=%s).",
        max_wait_steps,
        target_divisor,
    )


class SchedulerLoader(importlib.abc.Loader):
    def __init__(self, original_loader):
        self.original_loader = original_loader

    def create_module(self, spec):
        return self.original_loader.create_module(spec)

    def exec_module(self, module):
        self.original_loader.exec_module(module)
        patch_scheduler(module)

    def __getattr__(self, name):
        return getattr(self.original_loader, name)


class SchedulerFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "sglang.srt.managers.scheduler":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = SchedulerLoader(spec.loader)
        return spec


if os.environ.get("R1_ENABLE_LOCAL_PREFILL_BATCHER") == "1":
    sys.meta_path.insert(0, SchedulerFinder())

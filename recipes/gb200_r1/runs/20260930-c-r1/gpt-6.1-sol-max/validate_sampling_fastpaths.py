import json
from types import SimpleNamespace

import torch
import sgl_kernel
import sglang.srt.distributed as distributed
from r1_runtime import register
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.plugins.hook_registry import HookRegistry
from sglang.srt.server_args import set_global_server_args_for_scheduler
from sglang.srt.speculative import eagle_utils


torch.manual_seed(21)
register()
HookRegistry.apply_hooks()
set_global_server_args_for_scheduler(SimpleNamespace(
    speculative_use_rejection_sampling=True,
    speculative_accept_threshold_single=1.0,
    speculative_accept_threshold_acc=1.0,
))
distributed.get_tp_group = lambda: SimpleNamespace(world_size=1)
filter_calls = {'top_k': 0, 'top_p': 0}
original_top_k = sgl_kernel.top_k_renorm_prob
original_top_p = sgl_kernel.top_p_renorm_prob


def checked_top_k(*arguments, **parameters):
    filter_calls['top_k'] += 1
    return original_top_k(*arguments, **parameters)


def checked_top_p(*arguments, **parameters):
    filter_calls['top_p'] += 1
    return original_top_p(*arguments, **parameters)


sgl_kernel.top_k_renorm_prob = checked_top_k
sgl_kernel.top_p_renorm_prob = checked_top_p
results = []
vocabulary_size = 129280
for batch_size, slots, mode in ((1, 8, 'equal'), (4, 16, 'equal'),
                               (16, 4, 'equal'), (16, 12, 'equal'),
                               (16, 4, 'reject'), (4, 4, 'top_p'),
                               (4, 4, 'top_k')):
    filter_calls.update(top_k=0, top_p=0)
    logits = torch.randn(batch_size * slots, vocabulary_size, device='cuda') * 0.00003
    if mode == 'reject':
        logits[:, 0] = 6.0
    target = torch.softmax(logits / 0.3, dim=-1).reshape(batch_size, slots, -1)
    draft = target[:, :-1].contiguous()
    if mode == 'reject':
        draft = torch.full_like(draft, 1 / vocabulary_size)
    candidates = torch.zeros(batch_size, slots, dtype=torch.int64, device='cuda')
    candidates[:, 1:] = torch.multinomial(draft.flatten(0, 1), 1).reshape(batch_size, slots - 1)
    retrieve_index = torch.arange(batch_size * slots, device='cuda').reshape(batch_size, slots)
    empty_links = torch.full_like(retrieve_index, -1)
    sampling = SimpleNamespace(
        is_all_greedy=False,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        logit_bias=None,
        temperatures=torch.full((batch_size, 1), 0.3, device='cuda'),
        top_ks=torch.full((batch_size,), 5 if mode == 'top_k' else 1 << 30,
                          dtype=torch.int32, device='cuda'),
        top_ps=torch.full((batch_size,), 0.8 if mode == 'top_p' else 1.0, device='cuda'),
        need_top_k_sampling=mode == 'top_k',
        need_top_p_sampling=mode == 'top_p',
    )
    batch = SimpleNamespace(device='cuda', forward_mode=ForwardMode.DECODE,
                            seq_lens=torch.ones(batch_size, device='cuda'),
                            sampling_info=sampling)
    verify = SimpleNamespace(draft_token=candidates, draft_token_num=slots,
                             draft_probs=draft,
                             max_tree_depth=slots, tree_topk=1,
                             retrieve_index=retrieve_index,
                             retrieve_next_token=empty_links,
                             retrieve_next_sibling=empty_links)
    predicts, accepted, accept_index = eagle_utils.eagle_sample(
        verify, batch, SimpleNamespace(next_token_logits=logits)
    )
    if mode == 'equal':
        assert accepted.eq(slots).all().item(), accepted
    if mode == 'reject':
        assert accepted.eq(1).all().item(), accepted
        assert predicts[accept_index[:, 0]].eq(0).all().item()
    assert filter_calls == {'top_k': int(mode == 'top_k'), 'top_p': int(mode == 'top_p')}
    result = {'batch_size': batch_size, 'slots': slots, 'mode': mode,
              'accepted_tokens': accepted.tolist(), 'filter_calls': filter_calls.copy()}
    results.append(result)
    print(result, flush=True)
with open('sampling_fastpath_validation.json', 'w') as output:
    json.dump(results, output, indent=2)

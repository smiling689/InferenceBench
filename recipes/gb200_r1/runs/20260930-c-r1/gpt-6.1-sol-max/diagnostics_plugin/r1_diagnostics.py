import functools
import json
import os
from pathlib import Path


def register():
    if os.environ.get('R1_DIAGNOSTICS_ENABLED') != '1':
        return
    import torch
    from sglang.srt.speculative import reject_sampling

    original = reject_sampling.chain_speculative_sampling_triton
    recorded = {}
    call_count = 0

    @functools.wraps(original)
    def checked_sampling(*arguments, **parameters):
        nonlocal call_count
        call_count += 1
        batch_size = parameters['target_probs'].shape[0]
        inspect = recorded.get(batch_size, 0) < 4 and torch.distributed.get_rank() == 0
        if inspect:
            target = parameters['target_probs']
            draft = parameters['draft_probs']
            proposals = parameters['candidates'][:, 1:].long()
            proposal_target = target[:, :-1].gather(-1, proposals.unsqueeze(-1)).squeeze(-1)
            proposal_draft = draft.gather(-1, proposals.unsqueeze(-1)).squeeze(-1)
            ratios = proposal_target / proposal_draft.clamp_min(1e-30)
            statistics = {
                'call_count': call_count,
                'batch_size': target.shape[0],
                'slots': target.shape[1],
                'target_min': target.min().item(),
                'target_max': target.max().item(),
                'target_sum_min': target.sum(-1).min().item(),
                'target_sum_max': target.sum(-1).max().item(),
                'draft_min': draft.min().item(),
                'draft_max': draft.max().item(),
                'draft_sum_min': draft.sum(-1).min().item(),
                'draft_sum_max': draft.sum(-1).max().item(),
                'proposal_ratio_min': ratios.min().item(),
                'proposal_ratio_max': ratios.max().item(),
                'proposal_ratio_mean': ratios.mean().item(),
                'proposal_tokens': proposals[0].tolist(),
                'proposal_target': proposal_target[0].tolist(),
                'proposal_draft': proposal_draft[0].tolist(),
            }
        result = original(*arguments, **parameters)
        if inspect:
            statistics['accepted_drafts'] = parameters['accept_token_num'].tolist()
            output_path = Path(f'/home/agent/task/proposal_diagnostic.{os.getpid()}.jsonl')
            with output_path.open('a') as output:
                output.write(json.dumps(statistics) + '\n')
            recorded[batch_size] = recorded.get(batch_size, 0) + 1
        return result

    reject_sampling.chain_speculative_sampling_triton = checked_sampling

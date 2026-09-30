#!/usr/bin/env python3
"""Prepare a local SGLang overlay for FA4 prefill graph attention metadata.

The existing ROCm MHA companion selection also applies to the CUDA FA4 MHA
path. Only descriptor selection changes; all attention and model math remains.
Installed source files are read only and are never modified by this script.
"""
import difflib
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
source_root = Path('/sgl-workspace/sglang/python/sglang')
dest = root / 'graph_overlay' / 'modules'
dest.mkdir(parents=True, exist_ok=True)
changes = [
    ('srt/model_executor/model_runner.py',
     'if _is_hip and hasattr(layer.self_attn, "attn_mha"):',
     'if hasattr(layer.self_attn, "attn_mha"):'),
    ('srt/layers/radix_attention.py',
     'if _is_hip and not save_kv_cache and hasattr(attention_layer, "_pcg_mha_companion"):',
     'if not save_kv_cache and hasattr(attention_layer, "_pcg_mha_companion"):'),
    ('srt/model_executor/runner/prefill_cuda_graph_runner.py',
     '    def can_run_graph(self, forward_batch: ForwardBatch) -> bool:\n        if self._is_full_backend',
     '    def can_run_graph(self, forward_batch: ForwardBatch) -> bool:\n'
     '        # FA4 MHA captures only the no-prefix chunk path. Prefix hits need\n'
     '        # the ordinary prefill path to compute every cached-prefix chunk.\n'
     '        if (\n'
     '            self.model_runner.server_args.prefill_attention_backend == "fa4"\n'
     '            and any(forward_batch.extend_prefix_lens_cpu or ())\n'
     '        ):\n'
     '            return False\n'
     '        if self._is_full_backend'),
]
manifest = []
patch = []
for relative, before, after in changes:
    source = source_root / relative
    original = source.read_text()
    if original.count(before) != 1:
        raise RuntimeError('Installed SGLang source changed: ' + str(source))
    modified = original.replace(before, after)
    if source.name == 'prefill_cuda_graph_runner.py':
        replay_anchor = (
            '        self._prepare_forward_metadata_for_replay(\n'
            '            forward_batch, static_forward_batch, static_num_tokens\n'
            '        )'
        )
        replay_metadata = (
            '        # Captured MHA kernels do not replay Python field assignments.\n'
            '        # can_run_graph excludes prefix hits for this FA4 path.\n'
            '        if self.model_runner.server_args.prefill_attention_backend == "fa4":\n'
            '            for batch in (forward_batch, static_forward_batch):\n'
            '                batch.attn_attend_prefix_cache = False\n'
            '                batch.mha_return_lse = False\n'
            '                batch.mha_one_shot = False\n\n'
        )
        if modified.count(replay_anchor) != 1:
            raise RuntimeError('Prefill replay location changed: ' + str(source))
        modified = modified.replace(replay_anchor, replay_metadata + replay_anchor)
    target = dest / source.name
    target.write_text(modified)
    patch.extend(difflib.unified_diff(original.splitlines(True), modified.splitlines(True),
                                     fromfile=str(source), tofile=str(target)))
    manifest.append({'source': str(source), 'overlay': str(target),
                     'original_sha256': hashlib.sha256(original.encode()).hexdigest(),
                     'modified_sha256': hashlib.sha256(modified.encode()).hexdigest()})
(root / 'graph_overlay' / 'metadata_fix.patch').write_text(''.join(patch))
(root / 'graph_overlay' / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Prepared attention-metadata fixes and a prefix-cache guard in graph_overlay/')

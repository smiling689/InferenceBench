#!/usr/bin/env python3
"""Check captured prefill against eager prefill on independent synthetic tokens."""
import argparse
import json
import math
import random
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
url = 'http://127.0.0.1:30080'

def call(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url + path, data=data,
        headers={'Content-Type': 'application/json'} if data else {},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = response.read()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return body.decode()

rng = random.Random(7065)
prompts = [[rng.randrange(1000, 100000) for _ in range(n)] for n in (896, 960)]
payload = {
    'input_ids': prompts,
    'sampling_params': {'temperature': 0, 'max_new_tokens': 16, 'ignore_eos': True},
    'stream': False,
}
call('/flush_cache?timeout=10')
captured = call('/generate', payload)
call('/flush_cache?timeout=10')
# Requesting input logprobs intentionally selects SGLang's ordinary eager
# prefill path; the graph runner excludes start_len < extend_seq_len.
reference = call('/generate', {**payload, 'return_logprob': True, 'logprob_start_len': 0})
checks = []
for actual, expected in zip(captured, reference, strict=True):
    actual_ids = actual['output_ids']
    expected_ids = expected['output_ids']
    logprobs = expected['meta_info']['output_token_logprobs']
    checks.append({
        'same_output_ids': actual_ids == expected_ids,
        'completion_tokens': actual['meta_info']['completion_tokens'],
        'finite_reference_logprobs': all(math.isfinite(item[0]) for item in logprobs),
        'output_ids': actual_ids,
    })
passed = all(c['same_output_ids'] and c['completion_tokens'] == 16
             and c['finite_reference_logprobs'] for c in checks)
result = {'passed': passed, 'checks': checks, 'captured': captured, 'eager': reference}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'passed': passed, 'checks': checks}, indent=2))
if not passed:
    raise SystemExit('Graph/eager prefill equivalence check failed.')

# Final selection: C05, coherent bounded prefill coalescing

Canonical launcher: /home/agent/task/start_server.sh. Required implementation:
/home/agent/task/runtime_patches/r1_prefill_clock.py and sitecustomize.py.
Foreground exec of the installed SGLang server; all launches use test_server.sh
and the supervised scaffold. No alternate launcher or detached daemon.

## Selection evidence

Original full Scenario C, seed21, 256 requests per profile, unchanged traffic,
820–1023 target input/output tokens, temperature .3 and complete-output gate.
Objective is geometric mean of REQUEST throughput, not token throughput or
toolkit SLA. Held-out seed1337 is unavailable and has not been inspected.

| Candidate / full trial | Burst req/s | Poisson req/s | Constant req/s | Objective |
| --- | ---: | ---: | ---: | ---: |
| C01 / 0003 | 2.496797 | 1.834459 | 1.195278 | 1.762463 |
| C02 / 0005 | 2.684550 | 1.886264 | 1.225345 | 1.837571 |
| C03 / 0007 | 3.122310 | 2.113687 | 1.228104 | 2.008710 |
| C04 / 0009 | unavailable | unavailable | incomplete | INVALID |
| C05 / 0011 | 3.109190 | 2.182258 | 1.264194 | **2.047021** |
| C06 Tree / 0013 | 2.918471 | 2.128528 | 1.255223 | 1.982981 |

C05 improves measured objective 16.15% versus the initial baseline. All its
768 requested outputs complete, no errors/empty outputs or invalid profiles.
Single development runs do not establish statistical significance or accuracy.
Tree regresses 3.13% against C05, with every profile slower, and is restored
to default ordinary NCCL. Failed/unfinished trials are retained, not ranked.

## Configuration and correctness

- Full mounted deepseek-ai/DeepSeek-R1 at /models/deepseek-r1: 61 layers,
  256 routed experts, top-k 8, all hidden/head/intermediate dimensions unchanged.
- Native FP8 dummy weights; BF16 activations and latent KV. No weights/datasets
  downloaded, offload, external inference, cached responses or fabricated counts.
- TP4/EP1/DP1; TRT-LLM MLA, MoE and dense FP8 GEMM. Context cap4096, static
  memory .95, running cap128, prefill chunk4096, decode graphs 1/2/4/8/16/32/48/64;
  prefill graphs/radix cache disabled. These are serving limits, not model pruning.
- Queue ratio .15, cap350ms coalesces small admissions. Native per-rank wall
  clocks can choose different GPU phases; the task-local hook broadcasts one
  leader timeout bit over the existing CPU/Gloo control group and then executes
  the unchanged original scheduling/model path. Four-rank regression proves
  old divergence and repaired agreement for expired/unexpired leader clocks.
- All required safe-path settings remain: NCCL_NVLS_ENABLE=0,
  TORCH_SYMM_MEM_DISABLE_MULTICAST=1, enforce-disable-flashinfer-allreduce-fusion,
  disable-custom-all-reduce. Default NCCL algorithms/protocols, not forced LL128.
- Explicit Gloo loopback removes startup hostname delay. Server sampling seed
  327017732 materializes the measured effective C05 seed; native dummy weights
  independently use per-parameter seed1234. No seed sweep or accuracy claims.
- Launcher verifies hook files, creates task-local cache directories, loads the
  hook on every fresh process and execs foreground. It requires only the given
  installed dependencies/mounted model and files preserved under this task.

Capacity skill: about159.76 GiB weights/rank, 12.92 GiB BF16 KV/197376 tokens,
.42 GiB graphs; measured runtime peak183938 MiB. Limited remaining HBM and
small local batches argue against late replicated DP/EP or graph expansion.

## Toolkit and profiling evidence

Required benchmark/capacity/profiler/history skills read before configuration;
full toolkit helpers used by path. Relevant history retained in
history/model-pr-history-notes.md; installed 0.5.15.post1 checkout
0b3bb0cbe31873994c9f989fddfe2f87ca839fdd is the flag/source authority.
Profiler/pipeline evidence verifies complete 61-layer passes and nonzero GPU
events on every preserved rank. C05 fresh B64 prefill/decode and B16 warmed
decode use original workload lengths. B16 has122 MLA calls/rank (two complete
forwards), 4270 kernels, ~23.9ms span, with NCCL ~14–15% of overlapping sums.
Graph replay hides mapping; eager stack/shape prefill is perturbed source
evidence, not serving timing. Generic NSA catalog recommendations are rejected
for dense-MLA R1. Compute what-if is an estimate, not performance or MFU.

C04 incident bundle/replay and permission-denied native-stack attempts are
retained. Its inferred GPU-stall cause is not falsely called stack-confirmed.
C05 admission forwards reduce to353 with140 singleton admissions versus
C03 469/312 and C02 652/589. Ingress/detokenizer CPU samples show no streaming
bottleneck. This supports batching plus correctness repair, not a blind sweep.

## Fresh final validation

Selected launcher is restored under configs/c05_final_validation, including
exact log and implementation snapshots. Fresh quick trial_0014 and full
trial_0015 complete successfully: 12/12 and 768/768 outputs, no errors.
Final full confirmation objective **2.066720 request/s**, **+17.26%** versus initial baseline.
- burst: 3.145762 request/s.
- poisson: 2.203496 request/s.
- constant: 1.273523 request/s.
Fresh post-timing profiler mode both; actual remaining capture budget 121.6s, nonzero GPU events verified.
The original selected C05 score and this repeat are separate valid runs;
preview_metrics.json contains the final original full evaluator result.
The supervised foreground server remains at http://127.0.0.1:30080.
Final harness will kill and re-execute the standalone canonical script.

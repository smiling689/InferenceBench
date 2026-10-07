# DeepSeek-R1 Scenario C optimization

Budget: 2026-10-06 23:16:01 UTC to 2026-10-07 01:16:01 UTC.

Constraints: local complete 61-layer/256-expert DeepSeek-R1, FP8 dummy weights,
BF16 activations and KV cache, all four GB200 GPUs, no downloaded weights or datasets.
All launch configurations use start_server.sh in the foreground, ordinary NCCL,
NVLS disabled, multicast disabled, custom all-reduce and FlashInfer all-reduce fusion disabled.

Initial candidate: TP4, automatic Blackwell kernels, 0.93 static memory,
8192-token prefill chunks, maximum 128 running requests, decode CUDA graphs to 64.
Benchmark: 256 requests/profile; burst concurrency 64, poisson concurrency 32,
constant concurrency 16. Quick mode: 4 requests/profile. Development seed: 21.

Initial startup: full weights consume 159.6 GiB/GPU; BF16 MLA KV pool holds
143872 tokens/GPU (9.42 GiB). Decode CUDA graphs use 0.54 GiB/GPU.
Quick baseline: all 12 requests complete. Geometric mean 0.28464 requests/s;
first burst included cold-kernel compilation (TTFT median 12.71 s).
Full baseline is running for a meaningful high-load measurement.

Follow-up launch support: configurable kernels, EP/DP, streaming cadence,
mixed prefill/decode, and admission batching. All kernel caches are under
runtime_cache for reproducibility. Single-node control communication binds
loopback; inter-node IB is unnecessary. Fast DeepGEMM startup warmup preserves
the exact model computation and kernel dispatch.

## Full baseline

All 768 requests succeeded. Geometric mean: 1.756868 requests/s.
Burst: 2.502279; poisson: 1.823848; constant: 1.188210 requests/s.
Single-request refill prefills repeatedly block the entire decode batch.

Next candidate uses the original model's native MTP module as a speculative
drafter, with proper probabilistic rejection sampling (top-k=1), and decode
MLA for verification. Every accepted token is verified by the unchanged full
61-layer target, with all 256 experts and original top-k=8 routing intact.
No probability thresholds are relaxed and no responses are cached.
Also testing 8-token streaming chunks and 4-slot refill batching at max-run=64.

Native MTP-3 quick: 12/12 complete; geometric mean 0.793948 requests/s.
Warm poisson/constant TPOT median 3.60 ms (baseline 10.45 ms at four requests).
Observed accepted length is about 3.7 of 4 verified positions per iteration.
The target remains DeepseekV3ForCausalLM, weights 159.6 GiB/GPU; its native
NextN drafter adds 2.63 GiB/GPU. Both target and draft KV are BF16.
At memory fraction 0.93 the shared token capacity is 102080, so a higher
fraction and fewer CUDA graph buckets will be evaluated for larger windows.

The model generation_config has top_p=0.95. Next tests use documented OpenAI
sampling defaults (top_p=1.0 when the client omits it); all client-specified
sampling parameters remain honored. Rejection sampling remains exact.
SGLang's built-in GPU-worker CPU affinity is enabled for the 144-core node
(36 cores/GPU, matching its two 72-core CPU sockets).

## MTP-7 full confirmation

768/768 requests complete. Geometric mean: 3.860433 requests/s (2.1973x baseline).
Burst: 6.455989; poisson: 3.470947; constant: 2.567421 requests/s.
Configuration: TP4/EP1, memory 0.94, max-run 64, chunk 8192, stream interval 8,
four-slot admission batching, native MTP 7 steps/8 verification positions,
exact rejection sampling, OpenAI sampling defaults, CPU affinity,
decode graph buckets 1/2/4/8/16/32/48/64, automatic DeepGEMM dense kernels.
Accepted length is 8.00; each position is computed by the full target.
start_server.sh now defaults to this fully confirmed configuration.

Next experiment: 15 steps/16 positions, otherwise the same configuration.

MTP-15: quick requests completed, but accepted length collapsed to about 1.1
and decode throughput regressed severely. The subsequent 128-request screen
was stopped early and is NOT a valid measurement. MTP-7 remains the best
fully confirmed configuration. Next test adds bounded adaptive prefill
queue batching (ratio 0.25, maximum delay 250 ms/8 forward passes) to MTP-7.

## Adaptive prefill batching

128-request/profile screen: 384/384 complete, geometric mean 4.513512 req/s.
Full confirmation: 768/768 complete, geometric mean 4.382548 req/s.
Burst 5.422225; poisson 4.770067; constant 3.254452 req/s. This is 2.4945x
the full baseline and 13.5% above the previous fully confirmed best.
start_server.sh now defaults to the bounded adaptive prefill delayer as well.

## Further kernel and parallelism screens

FlashInfer TRTLLM dense FP8 GEMM (otherwise the same MTP-7 configuration)
completed 384/384 requests, geometric mean 4.096268 req/s, below the comparable
automatic/DeepGEMM screen of 4.513512. Dense FP8 therefore remains automatic.

Next screen: DP attention 4 and DP LM head, TP4/EP1, static memory 0.97,
local decode CUDA graph buckets 1/2/4/8/16 and global maximum 64 requests.
All original model dimensions and computation remain unchanged.

DP4 failed during dummy drafter initialization, before serving any requests.
The full target uses 173.24 GiB/GPU; replicated draft embeddings plus a
temporary FP8 initialization allocation exceed the 184 GiB usable capacity.
This is not a benchmark result. Next candidate uses DP2, memory fraction 0.96,
and local graph buckets 1/2/4/8/16/24/32.

DP2 screen completed 384/384: burst 6.206769, poisson 5.509421,
constant 1.277498, geometric mean 3.521901 req/s. Large local batches
accept almost all eight verification positions, while the eight-request
local graph (constant load) accepts about one. Next diagnostic keeps exact
rejection sampling and pads small batches to the already-valid 16-request
graph, using local buckets 16/24/32. No acceptance override is used.

DP2 with minimum graph batch 16 completed 384/384, but geometric mean
fell to 3.291954 req/s (burst 5.896020, poisson 5.147769, constant 1.175393).
Padding did not fix the constant-load acceptance regression. DP attention
therefore remains disabled in the persisted winner. Next candidate changes
only expert parallelism to EP4, keeping TP4 and ordinary NCCL collectives.

EP4 completed 384/384: geometric mean 4.123802 req/s (burst 5.082529,
poisson 4.622647, constant 2.984851), below the same-size TP4/EP1 winner.
The multicast all-gather correctly reported its ordinary NCCL fallback.
EP therefore remains 1. Next candidate uses FA4 prefill plus eager-compiler
piecewise prefill graphs at 1024/2048/4096 tokens, with memory fraction 0.93
to reserve graph workspace. Target decode/verification remains TRTLLM MLA.

FA4/pcg screen completed 384/384, geometric mean 3.806951 req/s. Burst
3.314879 includes substantial cold FA4 compilation; warm poisson reached
5.424909, constant 3.068116. SGLang explicitly disables tc_piecewise target
prefill graphs for EAGLE to prevent decode-replay corruption (#28386).
That safety guard is preserved. Next candidate uses the supported breakable
prefill graph backend and enables persistent FA4 CuTe DSL caching under
runtime_cache/fa4 for fresh-launch reproducibility.

FA4 plus breakable graphs failed during startup capture: the attention
backend attempted a 576-dimensional MLA view of a 192-dimensional query.
No requests were served and no valid score is claimed. Next graph candidate
keeps the original TRTLLM MLA attention backend for both prefill and decode,
with breakable prefill graphs at 1024/2048/4096 and static memory 0.93.

TRTLLM breakable capture failed with the same 192-versus-576 query-shape
mismatch. These prefill-graph candidates are rejected without changing
engine safety guards or model computation. Next candidate returns to the
fully confirmed TP4/EP1/MTP-7 configuration and increases bounded prefill
batching to 1000 ms / 32 forward passes (queue ratio remains 0.25).

## Final bounded-batching winner

The 128-request screen was lower (4.172989 req/s), but full confirmation
completed 768/768 and improved geometric mean to 4.488225 req/s:
burst 5.386017, poisson 4.901057, constant 3.425046. This is 2.5547x
the initial full baseline and 2.4% above the previous fully confirmed best.
The longer bounded delay is now persisted in start_server.sh. Final check
restarts through test_server.sh without experimental environment overrides,
then runs a quick smoke test and a full three-profile evaluation.

Observed low speculative acceptance at some small/padded CUDA graph shapes.
Next diagnostic tests the 16-token window with minimum graph batch 8
(buckets 8/16/32/48/64), keeping the target computation unchanged.

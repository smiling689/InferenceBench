# Final selection: highest valid measured scenario C candidate

Selected configuration: config01, full trial `trial_0003`. The original valid baseline remains the winner; no completed alternative demonstrates an improvement in the required geometric request-throughput objective. Measured improvement over that baseline: **0%**. This is a best-measured selection within the original two-hour budget, not a claim of a global optimum.

Final restoration verification: awaiting the original quick flow check in `trial_0014`; completion is recorded in `final_validation.json`.

## Full original evaluations

All successful full rows use the unchanged evaluator, development seed21,256 requests/profile, original820–1024 input/output-token distribution and burst/poisson/constant traffic. Each row completes768/768 requested outputs with zero failures. Quick flow checks are not used to rank performance. No latency SLA replaces the required objective.

| Configuration | Full trial | Burst req/s | Poisson req/s | Constant req/s | Geomean req/s | Decision |
|---|---|---:|---:|---:|---:|---|
| TP4/EP1, DeepGEMM dense | 0003 | 2.364544 | 1.848014 | 1.194326 | 1.734570 | Selected |
| DP4/EP4, memory fraction0.97 | 0007 | 1.285865 | 0.866126 | 0.631750 | 0.889421 | Rejected |
| TP4/EP1, TRT dense + stream16 | 0009 | 2.378935 | 1.771845 | 1.166125 | 1.700270 | Rejected |
| TP4 attention, EP4 routed | 0011 | 2.403142 | 1.768481 | 1.158273 | 1.701102 | Rejected |
| TP4/EP1, fixed8 NCCL channels | 0013 | 2.324415 | 1.754658 | 1.176413 | 1.686638 | Rejected |

Config02/DP4 at fraction0.94 fails startup with a negative KV budget: replicated dense/attention weights171.95GiB/rank exceed that allocation. Failed/aborted trials0001/0004/0005 remain in the evidence and are not ranked. Trial0001 also documents the initial evaluator-port mismatch; all subsequent evaluations explicitly use port30080.

## Standalone final launcher

`/home/agent/task/start_server.sh` foreground-execs installed SGLang0.5.15.post1 through the canonical `./test_server.sh` supervisor. No auxiliary daemon, cached response, external API, weight download or dataset download is used.

- Local full model/tokenizer `/models/deepseek-r1`; served name `deepseek-ai/DeepSeek-R1`.
- All61 layers,256 global routed experts, top-k8 and original dimensions are preserved. Dummy FP8 weights, BF16 model activations and BF16 KV; quality/accuracy is not evaluated.
- Four exclusive GB200 GPUs, each189471MiB; TP4/EP1 and original resolved seed951794684 explicitly pinned.
- TRTLLM MLA attention and routed MoE; DeepGEMM dense FP8; memory fraction0.94; context4096; max running64; prefill4096/max8192.
- Decode graph buckets1,2,4,8,16,24,32,48,64; prefill graphs off; radix cache off; stream interval1 (original default, now explicit).
- `NCCL_NVLS_ENABLE=0`, `TORCH_SYMM_MEM_DISABLE_MULTICAST=1`, `--enforce-disable-flashinfer-allreduce-fusion` and `--disable-custom-all-reduce` remain mandatory. Fixed-channel experimental overrides are removed; ordinary NCCL is used.

The pinning of the original effective seed and explicit default streaming value do not introduce a new performance configuration. The effective launch parameters match the selected full trial. `best/selected_start_server.sh` and `configs/config_01_final_start_server.sh` preserve this reproducible canonical launcher. The exact active-restoration log is `configs/config_01_final_server.log`.

## Evidence behind the choice

The selected capacity log reports approximately159.59GiB weights/rank,171648 BF16 KV tokens/11.24GiB,0.46GiB graphs and10.1GiB post-capture headroom. The64-request scoring load fits the logged KV budget, including2048-token requests. Actual physical HBM is taken from nvidia-smi, not the toolkit's inferred GPU label.

Baseline fresh traces in `profiles/config_01_bs64` preserve all four ranks and nonzero GPU events. Three complete61-layer decode forwards are verified with183 MLA anchors. Installed native routed GEMMs, DeepGEMM projections, MLA and ordinary NCCL actually execute; shared/routed dual-stream overlap and fused normalization already exist. The trace's graph replay attribution is not mistaken for the original caller.

DP4/EP4 suffers replicated projection cost and partial-prefill overhead; its CLI1024 prefill becomes256/rank in the installed implementation. TP4/EP4-only traces show unequal rank-local routed GEMM costs and complementary waits. Dense TRT kernels execute but do not beat DeepGEMM on the full objective. The fixed8-channel experiment really changes every observed collective grid from32 to8; extra KV capacity does not translate into a scoring improvement. All decisions use unprofiled full timing before fresh profiling.

Profiler windows requested as3 steps sometimes contain only2 complete forwards, which are counted explicitly. Instrumentation-induced rank-transition collective spikes are not presented as warmed serving latency. Kernel sums include stream overlap and are not equivalent to wall time. The evaluator's generation-token metric uses summed per-request decode time; streaming ITL is per chunk. Neither is substituted for request-throughput scoring.

The toolkit benchmark, capacity, profiler/fusion/overlap, pipeline/compute and DeepSeekV3/R1 PR-history evidence is retained in trial analyses, `history/model-pr-history-notes.md`, `results.md`, `toolkit_comparison.md` and rank-preserved profiles. Newer upstream flags/paths are not assumed supported by0.5.15.post1. The compute helper's defaultH20 label and partition assumptions are documented limitations; no MFU claim is made.

## Final validation and artifacts

The restored server is checked with the unchanged original quick mode,4 requests/profile, and native health/models endpoints. This is a restoration flow check, not a second full throughput claim. Its own metrics, capacity/log and per-request records are archived in `trials/trial_0014`.

`preview_metrics.json` and `best/selected_metrics.json` mirror the selected original full measurement after final smoke. Authoritative full per-request records are in `trials/trial_0003/generation_log_file.jsonl` and `requests_used_file.jsonl`, with archive hashes in `trial.json`; the restoration smoke remains separately archived. No counts or outputs are fabricated.

Original budget:2026-10-06 08:13:08–10:13:08 UTC. The final comparison and fresh four-rank capture complete before restoration. Another full run cannot fit the remaining budget without risking the required active-server state, so the measured winner is restored and flow-verified rather than promoting an unmeasured candidate. The held-out seed1337 is not accessed or optimized against. Final fresh-container scoring remains the harness's responsibility.

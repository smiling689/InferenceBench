# Final selection: full R1 TP4/EP1 with decode CUDA graphs

## Selected deployment

Canonical `./start_server.sh` sources root `server_config.env` and foreground
execs installed SGLang0.5.15.post1. Model/tokenizer load only the read-only
`/models/deepseek-r1` mount. The mounted architecture remains all61 layers,
hidden7168,128 heads, MLA latent512,256 routed experts, top8 and one shared
expert, with2048 expert intermediate size. Dummy FP8 weights, BF16
activations and BF16 KV are used. No weights/datasets are downloaded, no
model layer/expert computation is skipped, and no outputs are cached.
Accuracy/quality is not evaluated; no accuracy-equivalence claim is made.

Selected knobs: TP4/EP1; full decode graphs with buckets through64; normal
overlap scheduler; eager prefill chunks4096; concurrency64; KV131072 tokens;
context16384; mem_fraction_static0.94; page64; TRT-LLM MLA attention and
FlashInfer TRT-LLM FP8 MoE; radix cache disabled. Exact flags/config are
preserved in root scripts and config_002/. No installed source patch is
needed for final operation.

Rack safety is explicit in the launcher: NCCL_NVLS_ENABLE=0,
TORCH_SYMM_MEM_DISABLE_MULTICAST=1, enforce-disable-flashinfer-allreduce-fusion
and disable-custom-all-reduce. All collectives use the ordinary NCCL path;
startup reports FP8 deferred finalize disabled. No detached daemon is used.
Only test_server.sh supervises start_server.sh during development.

## Full unchanged Scenario C evidence

| Candidate / trial | Burst req/s | Poisson req/s | Constant req/s | Geomean req/s |
|---|---:|---:|---:|---:|
| TP4/EP1 decode graphs /0006 | 2.462059 | 1.829821 | 1.190994 | 1.750675 |
| TP4/EP4 decode graphs /0009 | 2.203789 | 1.645855 | 1.093373 | 1.582863 |

Both use original256 requests/profile, seed21,820–1024 input/output lengths,
traffic burst/poisson/constant and concurrency64/32/16. Each completes all768
requests with zero failures, empty outputs or incomplete generations. The
objective is the geometric mean of request throughput, not token throughput
or a toolkit-default SLA. EP4 is9.59% slower and loses every profile, so EP1
is restored exactly. Final evaluation's held-out1337 seed is not accessed.

The initial eager baseline0004 uses explicitly labeled development32
requests/profile (geomean0.139522). Restored002 trial0011 matches this same32
workload and completes all96 requests with geomean1.738097, approximately
12.46x higher throughput. This is a matched DEVELOPMENT comparison, not a
full256 speedup multiplier. Original quick checks use four/profile and
establish flow only, never high-load ranking.

## Bottleneck and choice evidence

Baseline eager trace shows TP0 host/rank-launch stalls with only3.66% GPU busy
union while other ranks largely wait in NCCL. Decode graphs plus scheduler
overlap remove that major bottleneck without reducing R1 computation. Full
002 reaches64 active requests without retraction; capacity skill reports
approximately159.6GiB weights,8.58GiB BF16 KV and0.87GiB graphs/rank, leaving
approximately12.3GiB immediately after graph capture. Runtime prefill buffer
growth still leaves HBM headroom. KV capacity is not the measured bottleneck.

Fresh002 decode32 and prefill16 traces follow unprofiled timing, preserve all
four ranks, use original full-length requests, and contain nonzero GPU events.
Decode TP0 sums~30.7ms over two full forwards; routed MoE GEMMs21.7%, attention
12.5%, NCCL13.5%. Eager prefill instead exposes host/rank collective waits
(519.9ms/80.8% kernel-duration share). These are duration shares, not removable
latency or wire-time estimates. Eager001 supplies compatible source mapping;
only fresh graph002 supplies graph timing.

Full-prefill graph003 fails startup: installed TRTLLMMLABackend delegates
EXTEND metadata to FlashInferMLABackend, which raises Invalid-mode EXTEND.
Accepted CLI and a generic prefill runner were insufficient support proof.
Failed quick0007 and exact log/source analysis are retained; no backend patch
or blind retry is attempted within the remaining budget.

EP4 experiment004 is source-verified before launch and fully timed. Fresh
decode32 and prefill16 afterward complete all48 probes and retain all ranks.
Pipeline analysis verifies61-layer anchor cadence and two complete forwards
plus a boundary fragment; requested profiler steps are not assumed to equal
GPU forwards. EP4 warm forward~14ms, extra separate shared MLP work and exposed
collective waits support its measured regression. First-pass cold-start
labels are helper heuristics, not evidence this post-benchmark trace was cold.
Graph replay scopes obscure source attribution; no EP1 MoE timings/mapping
are silently transferred. Unsafe catalog fusion suggestions remain excluded.

## Toolkit and reproducibility

Skills read and used from the full pinned `/opt/ai-infra-skills` tree:
llm-serving-auto-benchmark, llm-serving-capacity-planner,
llm-torch-profiler-analysis, model-pr-history-knowledge; pipeline/compute
skills are used only for specific architecture/timeline questions. Compute
what-if estimates are not benchmark results. Relevant PR4918/5571/5619/5977/
7371/9834 evidence and installed-source eligibility checks are retained in
history/model-pr-history-notes.md. Concrete flags and profiling routes are
verified against installed0.5.15.post1, not newer toolkit examples.

Every completed/failed trial has its own analysis.md, metrics, launch script,
effective server arguments, per-request evidence when available and exact-log
capacity analysis. comparison.md and skill_comparison.md group quick,
development and full separately. best_candidate.json preserves the best valid
full candidate; config_002 launcher/config are restorable. Original logs are
not reused for restored-instance capacity: current log is
config_002/restored_server.log.

Reproduce using root `./start_server.sh`, or use
`HOST=127.0.0.1 PORT=30080 ./test_server.sh` for the same supervision as final
evaluation. OpenAI GET /v1/models and streaming POST /v1/chat/completions are
native SGLang endpoints. Runtime kernel/autotune setup is performed by the
launcher; no detached process or cached response is required to survive.

## Restoration confirmation

Root config is byte-identical to measured002. Fresh supervised restoration
passes original quick0010: all12 complete, zero errors. Matched development32
0011 also completes all96, with the fair baseline delta documented above.
At10:04:21 only7.2minutes remain versus measured7.9minutes for a repeat full
before profiling/finalization; do not knowingly start an unfinished full.
Selection already has unchanged full0006 versus full0009 comparison. Larger
DEVELOPMENT64/trial0012 now validates burst64/poisson32/constant16 on the exact
restored instance, followed by fresh representative traces. This bounded
validation does not masquerade as a new full256 score; latest preview metrics
will be development64 and the full score remains archived in0006.

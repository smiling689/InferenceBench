# Sol/max with infra skills: two independent runs

Both independent experiments completed and passed the formal request audit.
Their throughput scores are 1.769123 and 1.746635 req/s. The mean is 1.757879
req/s and sample CV is 0.9046%; two observations do not establish stability.

| Node | Independent run | Start (Asia/Shanghai) | Planned optimization end |
| --- | --- | --- | --- |
| gb200-1 | 20261006-c-skills-r1 | 2026-10-06 16:13:08 | 18:13:08 |
| gb200-2 | 20261006-c-skills-r2 | 2026-10-06 16:11:35 | 18:11:35 |

Formal evaluation finished at 10:32:50 UTC on gb200-1 and 10:30:37 UTC on
gb200-2. Both verified all 768 held-out generation records with the exact
requested input/output token counts, full model shape, FP8 dummy weights,
BF16 activations/KV and ordinary NCCL. Effective formal arguments were
recovered from the startup logs; live endpoint metadata was not captured.

| Run | Burst req/s | Poisson req/s | Constant req/s | Geometric mean req/s |
| --- | --- | --- | --- | --- |
| skills r1 | 2.572436 | 1.807483 | 1.190846 | 1.769123 |
| skills r2 | 2.556156 | 1.775633 | 1.173993 | 1.746635 |

These values are below all three earlier ordinary Sol/max scores (6.454397,
2.130450 and 2.172502 req/s). Unequal small samples and the earlier ordinary
outlier prevent a strong treatment conclusion. The balanced four additional
runs per treatment are recorded in ../20261006-c-sol-skills-stability/.

Actual session tool calls include toolkit use in both runs (51 and 38 calls).
The first run has analysis files for all 14 in-budget evaluation starts; the
second has 11 for 12. Post-budget preview evaluations are counted separately.
These counts do not establish full loop compliance; the final notes and
representative profiler artifacts must be inspected separately. The raw
records remain in each node's results directory and the devbox results copy.

Two nodes each receive an independent two-hour `gpt-6.1-sol` / `max` run on
four GB200 GPUs. The treatment requires toolkit analysis after every completed
configuration experiment, then selection of the next configuration. Skill
reading, analysis and profiling consume the same optimization budget.

The original scenario C, dummy full DeepSeek-R1, development seed 21,
held-out seed 1337, request-throughput geometric mean and fresh-container
formal evaluation are retained. Agent tasks, Codex homes, sessions and final
images are independent. No previous optimized launcher or solution is given.

See [run_manifest.json](run_manifest.json) for immutable inputs and
run IDs, and [the treatment workflow](../../skill_guided/README.md) for the
implementation. Completed metrics, launchers and structured audits are in
each run's gpt-6.1-sol-max-skills/task/ directory.

Startup validation: both nodes matched 96 frozen runtime files and passed
18 original tests. The new wrapper preserved evaluation arguments, archived
immutable request artifacts and retained evaluator exceptions in its smoke
checks. Codex app-server actually discovered all eight selected skill entries.
Fixed request hashes match the earlier experiment. Each live agent has one
fresh session and has begun reading the benchmark, profiler, capacity and
model-history skill instructions.

Node controllers run under nohup independently of the interactive SSH session.
API forwarding is managed by persistent devbox user services with a six-hour
maximum lifetime. Credentials are provided over stdin, not saved in the
deployment or Docker configuration. Frozen treatment source: `b2c569f`.

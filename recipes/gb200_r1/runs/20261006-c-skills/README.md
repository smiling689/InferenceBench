# Sol/max with infra skills: two independent runs

Both independent experiments are running. The operator approved deployment,
and both Docker environments passed startup checks before their timers began.

| Node | Independent run | Start (Asia/Shanghai) | Planned optimization end |
| --- | --- | --- | --- |
| gb200-1 | 20261006-c-skills-r1 | 2026-10-06 16:13:08 | 18:13:08 |
| gb200-2 | 20261006-c-skills-r2 | 2026-10-06 16:11:35 | 18:11:35 |

Formal evaluation follows automatically and adds elapsed time. No final
performance score is available yet.

Two nodes each receive an independent two-hour `gpt-6.1-sol` / `max` run on
four GB200 GPUs. The treatment requires toolkit analysis after every completed
configuration experiment, then selection of the next configuration. Skill
reading, analysis and profiling consume the same optimization budget.

The original scenario C, dummy full DeepSeek-R1, development seed 21,
held-out seed 1337, request-throughput geometric mean and fresh-container
formal evaluation are retained. Agent tasks, Codex homes, sessions and final
images are independent. No previous optimized launcher or solution is given.

See [run_manifest.json](run_manifest.json) for immutable inputs and planned
run IDs, and [the treatment workflow](../../skill_guided/README.md) for the
implementation. Actual scores, skill-use evidence and timing will be recorded
after completion and compared with all three existing ordinary Sol/max runs.

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

# Four additional runs per treatment

Status: running. Two persistent node queues started on 2026-10-06 at
16:05:37 UTC (2026-10-07 00:05:37 Asia/Shanghai). Each starts an independent
gpt-6.1-sol/max run, with four GB200 GPUs, full FP8 dummy DeepSeek-R1,
scenario C and a 7200-second optimization budget. Analysis and profiling
consume that budget in the unchanged skills treatment.

| Queue position | gb200-1 | gb200-2 |
| --- | --- | --- |
| 1 | ordinary r4 | skills r4 |
| 2 | skills r3 | ordinary r5 |
| 3 | ordinary r6 | skills r6 |
| 4 | skills r5 | ordinary r7 |

Each node runs two repetitions of each treatment, using opposite alternating
orders. No launcher, previous score or agent session is reused. Held-out
formal evaluation uses seed 1337, 256 requests per profile and the unchanged
request-throughput geometric mean. Eight runs consume 16 optimization
node-hours, approximately eight elapsed optimization hours with two nodes;
post-solver previews and formal evaluations add time.

Both fresh CPU Docker preflights matched 96 frozen source files, verified
the two request hashes and no preinstalled task skills, and completed actual
Responses requests for gpt-6.1-sol with max reasoning. Node queues are detached
from SSH. Persistent API forwarding services have a 16-hour lifetime.

The final report will distinguish these eight new runs from the three earlier
ordinary and two earlier skills runs, include every repetition and any failure,
and audit request/token records, effective configurations and actual skills use.
Dummy weights support performance observations only; model quality is not
evaluated.

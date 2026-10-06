# Four additional runs per treatment

Status: running. Two persistent node queues started on 2026-10-06 at
16:05:37 UTC (2026-10-07 00:05:37 Asia/Shanghai). Each starts an independent
gpt-6.1-sol/max run, with four GB200 GPUs, full FP8 dummy DeepSeek-R1,
scenario C and a 7200-second optimization budget. Analysis and profiling
consume that budget in the unchanged skills treatment.

Six experiments have completed formal evaluation and each passed all 768
held-out generation records and exact token counts. Ordinary r4/r5/r6 score
2.103146/2.230331/1.965427 req/s; skills r3/r4/r6 score
2.912448/1.813369/1.998714 req/s. These are intermediate observations;
ordinary r7 and skills r5 are the last two planned experiments, now running.

Skills r6 selected a supported scheduler receive interval of 16 after
profiling and rejecting EP4/attention-DP4 candidates. Its 13 in-budget
evaluation starts have 12 analysis notes and 51 actual toolkit calls.
The final full confirmation lacks a separate analysis.md. Skills r3 likewise
lacks two final-confirmation notes (one evaluation crossed the deadline),
whereas skills r4 has all 12 in-budget notes. These gaps are retained;
installation, call counts and traces are not proof of complete loop compliance.

Installed-source checks distinguish runtime sampling seeds from dummy
initialization. The loader calls initialize_dummy_weights(model), which uses
the fixed per-parameter seed 1234; both base images and the six completed
final images have identical loader/initializer hashes. ServerArgs.random_seed
varies across selected launchers and can affect sampling/acceptance, rather
than the dummy initializer. See dummy_initialization_provenance.json.

The second pair's original post-budget preview timed out on readiness,
returning exit code 1. Fresh-container formal evaluation succeeded for both;
their two-hour optimization budgets were fully consumed. The preview failure
and extra elapsed time are retained in the run records.

Skills r3 selected native NextN with seven draft steps/eight target verification
slots and exact rejection sampling. Acceptance simulation is explicitly off.
The eagle_utils.py file appears changed in Docker metadata but its bytes match
the original image exactly; the experimental patch was restored. Native
formal logs report acceptance examples around 6.7–6.8 on these dummy weights.
See the run's implementation_audit.json. This is performance evidence in the
dummy setup, not an acceptance or accuracy claim for trained R1 weights.

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

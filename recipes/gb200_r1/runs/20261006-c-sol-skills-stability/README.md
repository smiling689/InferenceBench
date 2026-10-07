# Sol/max: four new repetitions per treatment

Completed all eight planned experiments: four ordinary and four skills-guided Sol 6.1/max runs. Recorded skills mean is 20.84% lower, with a lower median and substantial run-to-run variation. These samples do not demonstrate a stable large improvement.

The primary comparison retains all original-benchmark scores. Ordinary r7 selects OpenAI sampling defaults (omitted top_p=1.0), while the other seven inherit the model default top_p=0.95. This is an uncontrolled sampling-policy difference, so the mean gap is not a causal estimate of skills alone.

Each score is the formal geometric mean of burst/Poisson/constant request throughput, in req/s.
Every valid new score passed all 768 held-out generation records and exact token counts.

| Treatment | Valid / planned | Mean | Median | Sample CV | Range |
| --- | --- | --- | --- | --- | --- |
| plain | 4/4 | 2.747336 | 2.166738 | 47.32% | 1.965427–4.690439 |
| skills | 4/4 | 2.174752 | 1.986595 | 22.93% | 1.813369–2.912448 |

## Every new experiment

| Run | Node | Burst | Poisson | Constant | Score |
| --- | --- | --- | --- | --- | --- |
| 20261006-c-plain-r4 | gb200-1 | 3.281115 | 2.190150 | 1.294531 | 2.103146 |
| 20261006-c-plain-r5 | gb200-2 | 3.435663 | 2.309672 | 1.398129 | 2.230331 |
| 20261006-c-plain-r6 | gb200-1 | 3.096427 | 1.977469 | 1.239938 | 1.965427 |
| 20261006-c-plain-r7 | gb200-2 | 5.735870 | 5.195531 | 3.462671 | 4.690439 |
| 20261006-c-skills-r3 | gb200-1 | 3.919573 | 2.924996 | 2.154817 | 2.912448 |
| 20261006-c-skills-r4 | gb200-2 | 2.675950 | 1.846157 | 1.207013 | 1.813369 |
| 20261006-c-skills-r5 | gb200-1 | 2.986370 | 2.096452 | 1.229497 | 1.974476 |
| 20261006-c-skills-r6 | gb200-2 | 3.047801 | 2.093308 | 1.251504 | 1.998714 |

## Historical context

Historical runs remain separate from the balanced new four-versus-four comparison.

| Treatment | Earlier scores | Combined n | Combined mean | Combined sample CV |
| --- | --- | --- | --- | --- |
| plain | 6.454397, 2.130450, 2.172502 | 7 | 3.106670 | 56.70% |
| skills | 1.769123, 1.746635 | 6 | 2.035794 | 21.72% |

## Scope and limits

- Four new repetitions per treatment; descriptive comparison with small samples.
- Independent search outcomes include selected-configuration and runtime variation.
- Identical workload seeds; no claim of generalization to other traffic or lengths.
- Historical samples are shown separately and are not balanced by node or date.
- Skill artifact counts require manual interpretation; they do not prove complete compliance.
- Full dummy weights measure performance, not model quality.
- Runtime sampling seeds differ between runs and can affect generated tokens and native speculative acceptance; they are independent of the fixed per-parameter dummy initialization seed 1234.
- Native speculative acceptance on synthetic target/draft weights does not predict acceptance or throughput on trained DeepSeek-R1 weights.
- The original request payload omits top_p. Ordinary r7 selects OpenAI defaults (top_p=1.0); other runs select model defaults (top_p=0.95). The primary comparison retains every original-benchmark score, but it does not hold the effective sampling distribution fixed or isolate skills causally.

The original scenario C, FP8 dummy full DeepSeek-R1, BF16 activations/KV, four GB200 GPUs and 7200-second optimization budgets are retained.
Each node ran two ordinary and two skills runs in opposite alternating orders. Previews and fresh-container formal evaluation add elapsed time.
All eight optimization budgets consume 16 node-hours; the complete two-node pipeline elapsed 9.34 hours.
Skills were actually used in all four treatment runs. Three final-confirmation analysis files are missing across r3/r6; see workflow_audit.json for the recording gaps and source/profile review.
See comparison_summary.json for per-node statistics, effective configurations, timing and actual skill evidence; individual_results.csv preserves every new score.

## Skills and implementation audit

| Skills run | Actual toolkit calls | Analysis files / in-budget starts | Nonzero GPU traces |
| --- | ---: | ---: | ---: |
| r3 | 41 | 14/16 | 48 |
| r4 | 36 | 12/12 | 44 |
| r5 | 56 | 15/15 | 72 |
| r6 | 51 | 12/13 | 53 |

Trace totals include captures around final confirmation; they do not prove all captures completed inside the budget. Ordinary runs have zero toolkit calls, while some independently profile; zero skill-wrapper trials is not zero benchmark attempts.

Skills r3 retains native exact MTP rejection sampling; its experimental eagle_utils.py patch was restored byte-for-byte to the base image. Ordinary r7 also uses native seven-step/eight-slot MTP, with simulation disabled and unchanged target computation. Individual logged acceptance values on dummy weights are not trained-model acceptance estimates.

Skills r5 preserves a task-local scheduling hook under task/runtime_patches: it broadcasts the rank-leader timeout decision, then calls the original PrefillDelayer method. A fresh CPU Docker four-rank regression reproduces divergent original decisions and verifies unanimous patched decisions for both expired/unexpired cases, including timeout-threshold restoration. The hook is active in the successful formal launch. Required source files, regression and implementation_audit.json are retained with that run.

The two second-position runs' post-budget previews timed out after 1200 seconds of readiness waiting. Their optimization budgets were fully consumed and fresh-container formal evaluations succeeded; preview failure and extra elapsed time remain in the records.

## Additional descriptive context

Restricting to the model-default policy after inspecting configurations leaves three ordinary runs (mean 2.099635) and four skills runs (mean 2.174752). This post-hoc subset loses the planned equal counts and node balance; it is not a replacement for the primary four-versus-four result or a controlled effect estimate. See comparison_summary.json for its complete statistics.

All eight real Codex sessions show gpt-6.1-sol/max and distinct session hashes. The CPU Docker report cross-check passes all eight CSV/JSON scores against formal metrics and audited records (6144 total). Both base images and all eight final snapshots have matching loader/initializer hashes and per-parameter dummy seed 1234. Runtime sampling seeds remain independent and vary.

Frozen original runtime: 67a9cf389bd7568afac740e0e7cb8b6edcddc59b. Unchanged skills treatment: b2c569ffeadce33ab5187b59989097e421ed7390. Pinned toolkit: 6dc9c66a008daded66f214022919ff88b2186252. Both preflights matched 96 frozen source files and completed actual Sol/max Responses requests.

Both queues finished, all eight GPUs were inspected idle with no compute processes, and API forwarding services plus the observer were stopped. The unrelated development drafts were not deployed or included.

Full request/generation logs and Codex sessions remain in the node result directories; private devbox raw copies exclude sessions, traces and caches. Tracked result_audit.json files retain source hashes, effective arguments and actual call/trace evidence. run_manifest.json, sampling_contract_audit.json, dummy_initialization_provenance.json, workflow_audit.json and queue_state/ record protocol, limits and completion.

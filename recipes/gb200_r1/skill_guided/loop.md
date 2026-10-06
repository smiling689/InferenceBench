
## Skill-guided optimization treatment

Use the installed AI-Infra-Auto-Driven-SKILLS toolkit throughout this run.
This is one independent experiment, with no previous agent solution or score
provided. All skill reading, benchmarking, profiling, analysis, coding, server
startup, compilation and configuration decisions share the original two-hour
wall-clock optimization budget. The original Codex exec/resume solver controls
that budget. Do not ask for extra time or external human analysis.

The pinned toolkit is read-only at `/opt/ai-infra-skills`. These skills are
registered for Codex: `$llm-serving-auto-benchmark`,
`$llm-serving-capacity-planner`, `$llm-torch-profiler-analysis`,
`$llm-pipeline-analysis`, `$torch-profiler-layer-track`,
`$model-compute-simulation`, `$sglang-prod-incident-triage`, and
`$model-pr-history-knowledge`.

### Before the first configuration

Read the benchmark, profiler, capacity and history SKILL.md files. Read the
DeepSeek V3/R1 SGLang history and retain only relevant evidence in
`history/model-pr-history-notes.md` under your run directory. Consult the other
skills as needed. Verify concrete flags and profiling routes against the
installed SGLang 0.5.15.post1; the toolkit's newer upstream references are not
proof of support in this environment. Read paths and execute helpers from the
full toolkit tree so their relative references resolve.

### Required loop for every completed configuration experiment

1. State a configuration hypothesis and save the launch command and server
   log under your run directory. Launch through `./test_server.sh`. Do the
   original quick smoke first; complete or failed quick evaluations also need
   an analysis before a new configuration is selected. Run the unchanged
   scenario C evaluator for performance feedback. Never substitute the
   toolkit's example workloads, default SLA or default objective.
2. `evaluate.py` archives each evaluation's metrics, per-request records,
   launcher and effective server arguments in the trial directory printed at
   completion. It also invokes the capacity skill when a server log is
   available. Set `INFERENCE_BENCH_SKILL_SERVER_LOG` in the evaluation process
   to your current configuration's exact log path; never analyze an old log.
3. After EACH experiment, use the appropriate toolkit evidence to analyze its
   result before changing configuration. Write `analysis.md` in that trial
   directory: what was actually tested, failures/successes, three profile
   throughputs, timing limits, bottleneck evidence, skill files/scripts used,
   and the reason for the next configuration. Record changed knobs and why.
   If several evaluations belong to one configuration, link their trial IDs.
4. For a successful performance trial, capture a short fresh representative
   trace and run the profiler skill before choosing the next configuration.
   Profile AFTER unprofiled timing. Match scenario C input/output lengths
   (820–1024 tokens), observed batch/parallelism, and relevant prefill/decode
   phases; do not silently use the toolkit's generic 4090/1 or 1/2048 defaults.
   Bound the capture to a few steps, preserve ranks, and verify nonzero GPU
   events. When graph replay hides source context, distinguish mapping from
   warmed timing. If capture is unsupported, fails, or the remaining budget
   cannot cover it, record the concrete reason and analyze metrics/logs instead.
5. Look up relevant fusion/overlap families and model PR evidence. Decide the
   next configuration from observed bottlenecks rather than repeating a blind
   parameter sweep. Use pipeline/compute tools when they answer a specific
   unresolved question; their estimates are not performance measurements.
6. Keep the best valid measured launcher and required implementation artifacts
   restorable. Reserve time within the same deadline to restore that candidate
   and leave a working server. Summarize the evidence behind your final choice
   in `final_selection.md` under the run directory.

Quick mode remains four requests per traffic profile. It is a flow check and
cannot establish high-load throughput or reliably rank large batch limits.
Use full or explicitly labeled larger development evaluations when budget
permits. Failures consume budget and must remain in the evidence.

The scenario, inputs, GPU allocation, FP8 dummy weights, BF16 activations/KV,
model dimensions and scoring contract above take precedence over toolkit
defaults. Download neither weights nor datasets. Dummy runs do not support
accuracy claims; do not fabricate acceptance rates or bypass computation.
The held-out seed 1337 is unavailable during optimization. Your final score
will be obtained by the unchanged fresh-container evaluation after the solver
ends, and is not an opportunity for further optimization.

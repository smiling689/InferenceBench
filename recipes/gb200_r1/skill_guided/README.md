# Skill-guided Sol/max comparison

Run two independent `gpt-6.1-sol` / `max` experiments, one per node, each on
four GB200 GPUs with a 7200-second optimization budget. Skill use is the
treatment: after each configuration experiment, the agent analyzes evidence
with the pinned toolkit before choosing the next configuration.

The toolkit is pinned to `BBuf/AI-Infra-Auto-Driven-SKILLS` commit
`6dc9c66a008daded66f214022919ff88b2186252`. Seven relevant task skills plus
`model-pr-history-knowledge` are registered in the isolated container's user
skill directory. New-model onboarding, architecture pictures and PR review
skills are excluded from this serving experiment. Preserve the toolkit's full
relative layout, including `docs/` and the PR history tree.

The underlying benchmark runtime remains the previously deployed
`67a9cf389bd7568afac740e0e7cb8b6edcddc59b`: original Codex exec/resume,
scenario C, fixed development requests (seed 21), separate final requests
(seed 1337), FP8 dummy weights, BF16 activations/KV and ordinary NCCL.
The scenario evaluator and final scoring are unchanged. No prior agent
solution, score, conversation, compiled solution image or task is provided.

`agent.py` calls the existing setup and solver, registers the skills and
appends [loop.md](loop.md). The development `evaluate.py` wrapper adds
per-trial evidence snapshots and a capacity-analysis helper outside benchmark
timing; both consume the existing optimization budget. The agent performs
profiler/history analysis and writes its next-configuration rationale.
Skill compliance must be checked from the actual session and trial artifacts,
not inferred from installation alone.

`run.sh` launches the agent and then snapshots its container for the existing
fresh-container held-out evaluation. Optimization receives only the seed-21
request file. Each run uses a new task, Codex home, container and result path.
Startup, the original post-solver preview and held-out evaluation can add
wall-clock time beyond the configured two-hour optimization budget.

The toolkit is staged outside Git under
`results/gb200_r1/20261006-c-skills-preparation/vendor/`. No credentials belong
there. Credential JSON is supplied over stdin; a separate SSH reverse forward
can expose the API on loopback when nodes cannot reach it directly.

Before starting, inspect resources, verify frozen runtime hashes and request
hashes, run `run.sh inspect RUN_ID` inside the node's Docker environment and
check Codex skill discovery/API access. Synchronization requires the operator's
explicit approval after a remote inspection and rsync dry-run.

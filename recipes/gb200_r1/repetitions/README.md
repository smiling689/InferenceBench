# Independent Sol/max repetitions

Run four additional ordinary and four additional skill-guided experiments.
Each fresh container receives the existing scenario C prompt, development
seed 21, gpt-6.1-sol/max and a 7200-second optimization budget. Skill runs use
the unchanged treatment from b2c569f and pinned toolkit 6dc9c66. Ordinary runs
have no toolkit mount or registered task skills. All runs begin with empty
task and Codex directories and receive no earlier solution or score.

The schedule alternates treatments, with opposite ordering on the two nodes.
Each node executes two ordinary and two skill runs, sequentially using all
four GPUs. Eight runs require 16 optimization node-hours, approximately eight
wall-clock hours with two nodes; previews, commits and formal evaluations add
time. Formal scoring retains the original held-out seed 1337 and fresh-image
evaluator. The metadata observer sends no generation traffic.

Preflight and API connectivity are checked before starting either queue.
Run queue.sh with a node name and one private JSON credential line on stdin.
The credential stays in process memory and is passed to each agent over stdin.
Queue locks and existing-output checks prevent duplicate launches. Preserve
failed experiments in completed.tsv; do not replace them silently.

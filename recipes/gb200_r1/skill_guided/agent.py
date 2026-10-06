#!/usr/bin/env python3
"""Add pinned infra skills to the existing GB200 Codex experiment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = Path("/opt/ai-infra-skills")
SKILLS_COMMIT = "6dc9c66a008daded66f214022919ff88b2186252"
SKILL_NAMES = (
    "llm-serving-auto-benchmark", "llm-serving-capacity-planner",
    "llm-torch-profiler-analysis", "llm-pipeline-analysis",
    "torch-profiler-layer-track", "model-compute-simulation",
    "sglang-prod-incident-triage",
)


def load_experiment():
    spec = importlib.util.spec_from_file_location("gb200_original", HERE.parent / "experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def register_skills() -> dict:
    registration = Path.home() / ".agents/skills"
    registration.mkdir(parents=True, exist_ok=True)
    paths = {name: VENDOR / "skills" / name for name in SKILL_NAMES}
    paths["model-pr-history-knowledge"] = VENDOR / "model-pr-optimization-history"
    manifest = {}
    for name, target in paths.items():
        skill = target / "SKILL.md"
        if not skill.is_file():
            raise FileNotFoundError(skill)
        link = registration / name
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise RuntimeError(f"Unexpected existing skill: {link}")
        elif link.exists():
            raise RuntimeError(f"Refusing to replace existing skill: {link}")
        else:
            link.symlink_to(target, target_is_directory=True)
        manifest[name] = {"path": str(target), "skill_sha256": hashlib.sha256(skill.read_bytes()).hexdigest()}
    return manifest


def setup_agent(experiment, original_setup, model: str, hours: float) -> None:
    original_setup(model, hours)
    task = experiment.TASK
    run_dir = task / "agent" / ("run_" + experiment.dt.datetime.now(experiment.dt.timezone.utc).strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True)
    os.environ["INFERENCE_BENCH_SKILL_TRIALS"] = str(run_dir / "trials")
    manifest = {
        "skills_repo": "https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS",
        "skills_commit": SKILLS_COMMIT,
        "registered_skills": register_skills(),
        "baseline_runtime_commit": "67a9cf389bd7568afac740e0e7cb8b6edcddc59b",
        "optimization_budget_seconds": int(hours * 3600),
        "reasoning_effort": "max", "run_directory": str(run_dir),
        "independent_start": True, "quality_evaluated": False,
    }
    experiment.save(task / "skill_manifest.json", manifest)
    prompt_path = Path(os.environ["PROMPT_FILE"])
    prompt = prompt_path.read_text() + "\n" + (HERE / "loop.md").read_text()
    prompt += f"\nYour experiment evidence directory is `{run_dir}`.\n"
    prompt_path.write_text(prompt)
    (task / "prompt.txt").write_text(prompt)
    (task / "evaluate.py").write_text(
        "#!/usr/bin/env python3\n"
        "from recipes.gb200_r1.skill_guided.feedback import main\n"
        "main()\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["gpt-6.1-sol"], default="gpt-6.1-sol")
    parser.add_argument("--hours", type=float, default=2)
    args = parser.parse_args()
    if args.hours != 2:
        parser.error("This comparison fixes the optimization budget at two hours")
    experiment = load_experiment()
    original_setup = experiment.setup_agent
    experiment.setup_agent = lambda model, hours: setup_agent(experiment, original_setup, model, hours)
    experiment.agent(args)


if __name__ == "__main__":
    main()

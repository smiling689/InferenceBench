from src.eval.inference import hpo_search_baselines as hpo
from src.eval.inference import runner


def test_chat_template_mapping_counts_ids_not_fields():
    from collections import UserDict
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return UserDict(input_ids=[1, 2, 3, 4, 5], attention_mask=[1] * 5)
    assert runner._count_chat_tokens([{"role": "user", "content": "hi"}], Tokenizer()) == 5


def test_failed_requests_cannot_win_dummy_performance_search():
    metrics = {"profiles": {"burst": {"request_count": 4, "success_count": 0,
               "request_throughput_req_per_s": 1000000}}}
    metrics["performance_check"] = runner.performance_check(metrics)
    score, reason = hpo.primary_metric(metrics, "inference_scenario_c_high_load")
    assert score == hpo.FAILURE_SCORE
    assert reason == "incomplete_requests"


def test_dummy_final_keeps_quality_disabled():
    env = hpo.build_trial_env("sglang", {}, {"INFERENCE_BENCH_PERFORMANCE_ONLY": "1"}, skip_quality=False)
    assert env["INFERENCE_BENCH_SKIP_QUALITY"] == "1"
    assert not hpo.gate_passed({"quality_check": {"evaluated": False, "pass": None}})


def test_dummy_evaluation_never_runs_quality(monkeypatch, tmp_path):
    monkeypatch.setenv("INFERENCE_BENCH_PERFORMANCE_ONLY", "1")
    monkeypatch.setattr(runner, "run_speed_eval", lambda *args: {"profiles": {"burst": {"request_count": 4, "success_count": 4}}})
    def forbidden(*args):
        raise AssertionError("Dummy evaluation attempted to run accuracy evaluation")
    monkeypatch.setattr(runner, "run_quality_eval", forbidden)
    args = runner.build_parser().parse_args(["--json-output-file", str(tmp_path / "metrics.json")])
    result = runner.run_evaluation(tmp_path, args)
    assert result["quality_evaluated"] is False
    assert result["quality_check"]["pass"] is None
    assert result["performance_check"]["pass"] is True


def test_poisson_replay_reproducible_only_in_performance_mode(monkeypatch):
    monkeypatch.setenv("INFERENCE_BENCH_PERFORMANCE_ONLY", "1")
    monkeypatch.setenv("INFERENCE_BENCH_DATASET_SEED", "21")
    first = runner._schedule("poisson", 256, 32)
    assert first == runner._schedule("poisson", 256, 32)
    monkeypatch.setenv("INFERENCE_BENCH_DATASET_SEED", "1337")
    assert first != runner._schedule("poisson", 256, 32)


def test_reasoning_tokens_are_visible_in_dummy_r1_stream(monkeypatch):
    monkeypatch.setenv("INFERENCE_BENCH_PERFORMANCE_ONLY", "1")
    chunk = {"choices": [{"delta": {"content": None, "reasoning_content": "thinking"}}]}
    assert runner._extract_delta_text(chunk) == "thinking"

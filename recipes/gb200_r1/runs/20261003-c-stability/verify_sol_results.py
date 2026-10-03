import collections, hashlib, json, math, sys
from pathlib import Path
task=Path("/home/agent/task")
metrics_path=task/"final_metrics.json"
metrics=json.loads(metrics_path.read_text())
source=Path("/bench-inputs/requests_1337.jsonl")
expected_hash="08fb8644ae4fee2939187086103a9a310866be4b9bd5f50b9c2aef2a343d6cae"
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected_hash
expected=[json.loads(line) for line in source.read_text().splitlines()]
used_path=Path(metrics["speed_eval"]["requests_used_file"])
used=[json.loads(line) for line in used_path.read_text().splitlines()]
assert len(expected)==len(used)==256
fields=("sample_id","messages","max_new_tokens","temperature","ignore_eos","input_token_count")
for a,b in zip(expected,used):
    assert all(a.get(k)==b.get(k) for k in fields),(a["sample_id"],"request differs")
    assert b["temperature"]==0.3 and b["ignore_eos"] is True
    assert 820<=b["input_token_count"]<=1024 and 820<=b["max_new_tokens"]<=1024
assert metrics["speed_eval"]["requests_source"]==str(source)
assert metrics["model_id"]=="deepseek-ai/DeepSeek-R1"
profiles=metrics["profiles"]
assert set(profiles)=={"burst","poisson","constant"}
assert metrics["performance_check"]["pass"] is True and not metrics.get("error")
for p in profiles.values():
    assert p["request_count"]==p["success_count"]==256
    assert p["failure_count"]==p["empty_output_count"]==0
    assert p["request_throughput_req_per_s"]>0
generation_path=Path(metrics["speed_eval"]["generation_log_file"])
rows=[json.loads(line) for line in generation_path.read_text().splitlines()]
assert len(rows)==768
for block in range(3):
    batch=rows[block*256:(block+1)*256]
    assert {r["request_index"] for r in batch}==set(range(256))
    for r in batch:
        req=used[r["request_index"]]
        assert r["sample_id"]==req["sample_id"]
        assert r["success"] is True and not r["empty_output"] and not r["error"]
        assert isinstance(r["model_output"],str) and r["model_output"].strip()
        assert r["output_tokens"]==r["tokens"]==req["max_new_tokens"]
        assert r["input_tokens"]==req["input_token_count"]
model_path=Path("/models/deepseek-r1/config.json")
model=json.loads(model_path.read_text())
shape={k:model[k] for k in ("num_hidden_layers","hidden_size","n_routed_experts","num_experts_per_tok","num_attention_heads")}
assert shape=={"num_hidden_layers":61,"hidden_size":7168,"n_routed_experts":256,"num_experts_per_tok":8,"num_attention_heads":128}
info=json.loads((task/"formal_server_info.json").read_text())
args=info.get("server_args",info)
assert args["load_format"]=="dummy" and args["quantization"]=="fp8"
assert args["dtype"]=="bfloat16" and args["kv_cache_dtype"] in ("bf16","bfloat16")
assert args["tp_size"]==4
score=math.exp(sum(math.log(p["request_throughput_req_per_s"]) for p in profiles.values())/3)
summary=json.loads((task/"summary.json").read_text())
assert math.isclose(score,summary["primary_metric"],rel_tol=1e-12)
result={"audit_passed":True,"official_evaluation_seed":1337,"request_source_sha256":expected_hash,
"requests":256,"profiles":3,"generation_records":len(rows),"complete_success_records":len(rows),
"input_output_token_counts_exact":True,"request_fields_unchanged":list(fields),"model_shape":shape,
"model_config_sha256":hashlib.sha256(model_path.read_bytes()).hexdigest(),
"weights_format":"fp8 dummy","activation_dtype":"bfloat16","kv_dtype":args["kv_cache_dtype"],
"primary_metric":score,"profile_throughputs":{k:v["request_throughput_req_per_s"] for k,v in profiles.items()},
"total_requested_output_tokens_per_profile":sum(r["max_new_tokens"] for r in used),
"generation_log_sha256":hashlib.sha256(generation_path.read_bytes()).hexdigest(),
"requests_used_sha256":hashlib.sha256(used_path.read_bytes()).hexdigest(),"quality_evaluated":False}
(task/"result_audit.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))

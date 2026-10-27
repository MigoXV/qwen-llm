import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.performance.qwen38.cli import app
from tests.performance.qwen38.config import ProfileConfig
from tests.performance.qwen38.analysis import analyze_directory, analyze_round, analyze_trace, interval_union
from tests.performance.qwen38.runner import ProfileSuite, config_from_catalog, scenarios, summarize_round


def test_interval_union_avoids_double_counting_parallel_and_nested_work():
    assert interval_union([(1, 5), (2, 3), (4, 8), (10, 12)]) == 9
    assert interval_union([]) == 0


def test_trace_separates_host_time_from_parallel_device_work(tmp_path):
    path = tmp_path / "trace_view.json"
    path.write_text(json.dumps({"traceEvents": [
        {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "python"}},
        {"ph": "M", "name": "process_name", "pid": 2, "args": {"name": "Ascend Hardware"}},
        {"ph": "X", "name": "qwen_step/0/decode", "pid": 1, "ts": 0, "dur": 500},
        {"ph": "X", "name": "WeightQuantBatchMatmul", "cat": "Kernel", "pid": 2, "ts": 10, "dur": 100},
        {"ph": "X", "name": "other", "cat": "Kernel", "pid": 2, "ts": 60, "dur": 100},
    ]}))
    result = analyze_trace(path)
    assert result["device_work_us"] == 200
    assert result["device_busy_union_us"] == 150
    assert result["device_idle_within_span_us"] == 0
    assert result["groups_us"]["FFN INT4 antiquant matmul"] == 100
    assert result["host_scopes"]["qwen_step/decode"]["host_wall_us"] == 500


def test_empty_device_trace_cannot_be_accepted_as_model_profile(tmp_path):
    path = tmp_path / "trace_view.json"
    path.write_text('{"traceEvents": []}')
    assert not analyze_trace(path)["valid_device_trace"]


def test_ascend_string_timestamps_and_wait_tasks_are_not_compute(tmp_path):
    path = tmp_path / "trace_view.json"
    path.write_text(json.dumps([
        {"ph": "M", "name": "process_name", "pid": 2, "args": {"name": "Ascend Hardware"}},
        {"ph": "X", "name": "EVENT_WAIT", "pid": 2, "ts": "1792643510000000.001", "dur": 90,
         "args": {"Task Type": "EVENT_WAIT"}},
        {"ph": "X", "name": "Matmul", "pid": 2, "ts": "1792643510000090.001", "dur": 0.1,
         "args": {"Task Type": "AI_CORE"}},
    ]))
    result = analyze_trace(path)
    assert result["device_work_us"] == 0.1
    assert result["device_control_accumulated_us"] == 90
    assert result["device_busy_union_us"] == pytest.approx(0.1)


def test_async_dequeue_bridge_attributes_device_work_to_originating_phase(tmp_path):
    path = tmp_path / "trace_view.json"
    path.write_text(json.dumps([
        {"ph": "M", "name": "process_name", "pid": 2, "args": {"name": "Ascend Hardware"}},
        {"ph": "X", "name": "qwen_step/0/prefill", "pid": 1, "tid": 1, "ts": 100, "dur": 20},
        {"ph": "X", "name": "Enqueue@op", "pid": 1, "tid": 1, "ts": 102, "dur": 1,
         "cat": "enqueue", "args": {"correlation_id": 3}},
        {"ph": "X", "name": "Dequeue@op", "pid": 1, "tid": 4, "ts": 130, "dur": 3,
         "cat": "dequeue", "args": {"correlation_id": 3}},
        {"ph": "s", "name": "HostToDevice", "pid": 5, "tid": 4, "ts": 131,
         "cat": "HostToDevice", "id": str(5 << 32)},
        {"ph": "X", "name": "Matmul", "pid": 2, "tid": 6, "ts": 200, "dur": 10,
         "args": {"Task Type": "AI_CORE", "Model Id": 4294967295, "connection_id": 5}},
    ]))
    result = analyze_trace(path)
    assert result["attribution"]["phase_operators_us"]["prefill"]["Matmul"] == 10


def test_ablation_preserves_context_cache_and_model_when_changing_mtp(tmp_path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps([{"name": "target", "path": "/models/test", "kv_cache_gib": 12.5,
                                    "max_input_tokens": 57344, "batch_tokens": 8192, "speculative_tokens": 2}]))
    baseline = config_from_catalog(catalog, "target")
    ablation = config_from_catalog(catalog, "target", speculative_tokens=0)
    assert baseline.speculative_tokens == 2 and ablation.speculative_tokens == 0
    assert baseline.max_input_tokens == ablation.max_input_tokens == 57344
    assert baseline.kv_cache_bytes == ablation.kv_cache_bytes == int(12.5 * 1024**3)
    assert baseline.model_path == ablation.model_path


def test_long_cases_respect_capacity_and_include_mixed_when_supported():
    small = scenarios(ProfileSuite.FULL, 8192)
    assert max(c.get("input_tokens", 0) for c in small) == 8192
    large = scenarios(ProfileSuite.FULL, 57344)
    assert any(c["name"] == "mixed32k-c4" for c in large)
    assert any(c.get("input_tokens") == 57344 for c in large)


def test_failed_requests_remain_in_makespan_and_invalidate_round():
    client = {"samples": [
        {"start_ns": 0, "end_ns": 1_000_000_000, "first_text_ns": 100_000_000,
         "response": {"usage": {"input_tokens": 38, "output_tokens": 10,
                      "input_tokens_details": {"cached_tokens": 0}}}},
        {"start_ns": 0, "end_ns": 2_000_000_000, "error": "failed"},
    ]}
    result = summarize_round(client, {})
    assert not result["valid"]
    assert result["tokens_per_second"] == 5


def test_mtp_decode_time_uses_generated_tokens_not_delivery_count():
    frontend = {"preparations": [], "deliveries": [
        {"request_id": "r", "submitted_ns": 0, "at_ns": 100_000_000, "new_tokens": 1, "total_tokens": 1},
        {"request_id": "r", "submitted_ns": 0, "at_ns": 160_000_000, "new_tokens": 3, "total_tokens": 4},
        {"request_id": "r", "submitted_ns": 0, "at_ns": 220_000_000, "new_tokens": 3, "total_tokens": 7},
    ]}
    result = analyze_round({"samples": [{"response": {"id": "r"}, "end_ns": 230_000_000}]},
                           frontend, {"steps": [], "schedules": []})
    request = result["requests"][0]
    assert request["decode_ms_per_token"] == 20
    assert request["batch_tokens"] == {1: 1, 3: 2}
    assert request["engine_batch_gap_ms_median"] == 60
    assert request["engine_batch_gap_ms_max"] == 60


def test_profile_cli_rejects_graph_module_hooks_before_occupying_device(tmp_path):
    result = CliRunner().invoke(app, ["profile", "--output-dir", str(tmp_path / "run"), "--module-ranges"])
    assert result.exit_code != 0
    assert not (tmp_path / "run").exists()


def test_profile_rejects_unknown_hardware_counter():
    with pytest.raises(ValueError):
        ProfileConfig(Path("/tmp/profile"), aic_metrics="typo")


def test_offline_report_retains_partial_failed_case_without_worker_artifact(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"config": {}, "cases": [{"name": "failed", "rounds": []}]}))
    case = tmp_path / "failed"
    case.mkdir()
    (case / "round0-http.json").write_text('{"samples": [{"index": 0, "error": "worker crashed"}]}')
    result = analyze_directory(tmp_path)
    partial = result["cases"][0]["details"]["round0"]
    assert not partial["valid"]
    assert "worker.json" in partial["incomplete_artifacts"]
    assert (tmp_path / "analysis.json").exists()


@pytest.mark.parametrize("backend,operator", [
    ("upstream", "WeightQuantBatchMatmul"), ("gmm", "GroupedMatmul")])
def test_partial_device_trace_is_detected_against_actual_forward_count(tmp_path, backend, operator):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "config": {"w4a8_linear_backend": backend},
        "cases": [{"name": "short128-c1", "rounds": []}]}))
    (tmp_path / "modules.json").write_text(json.dumps([[{
        "role": "target", "name": "model.layers.0.mlp.down_proj",
        "parameters": {"weight": {"dtype": "torch.int32"}}}]]))
    case = tmp_path / "short128-c1"
    case.mkdir()
    (case / "trace-worker.json").write_text(json.dumps([{"steps": [
        {"scheduled_tokens": {"request": 1}}, {"scheduled_tokens": {"request": 1}}]}]))
    trace = tmp_path / "traces/short128-c1-trace_rank0/ASCEND_PROFILER_OUTPUT"
    trace.mkdir(parents=True)
    (trace / "trace_view.json").write_text(json.dumps([
        {"ph": "X", "name": operator, "cat": "Kernel",
         "pid": 2, "ts": 10, "dur": 100}]))
    result = analyze_directory(tmp_path)["traces"][0]
    assert result["valid_device_trace"]
    assert result["coverage"] == {"target_forwards": 2, "w4_layers": 1,
        "expected_matmuls": 2, "observed_matmuls": 1, "complete": False}

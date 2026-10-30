from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from importlib.metadata import version
import sys

import typer

from qwen_llm.config import GraphMode, ModelAdaptation, ModelDType, Runner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Qwen LLM 命令行工具。", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Qwen LLM 项目工具。"""


@app.command("version")
def show_version() -> None:
    """显示已安装的项目版本。"""
    typer.echo(version("qwen-llm"))


@app.command("download-model")
def download_model(
    model_id: str = typer.Argument(..., envvar="QWEN_MODELSCOPE_ID"),
    output_dir: Path = typer.Option(Path("model-bin/Qwen3-14B-W8A8-Ascend"), envvar="QWEN_DOWNLOAD_DIR"),
    workers: int = typer.Option(4, min=1, max=8, envvar="QWEN_DOWNLOAD_WORKERS"),
) -> None:
    """从 ModelScope 下载并校验公开模型文件；支持断点续传。"""
    from qwen_llm.modelscope import download_modelscope
    download_modelscope(model_id, output_dir, workers)


@app.command("serve-model", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def serve_model(
    ctx: typer.Context,
    name: str = typer.Argument("qwen3.8-27b-mtp2", envvar="QWEN_MODEL_PROFILE"),
    catalog: Path = typer.Option(Path(__file__).resolve().parents[3] / "configs/models.json",
                                 exists=True, dir_okay=False, envvar="QWEN_MODEL_CATALOG"),
) -> None:
    """按模型配置选择权重与已验证参数；后面的serve选项可覆盖配置。"""
    entries = json.loads(catalog.read_text())
    row = next((item for item in entries if item["name"].casefold() == name.casefold()), None)
    if row is None:
        raise typer.BadParameter("未知模型；可选：" + ", ".join(item["name"] for item in entries))
    for field, supported in (("max_output_tokens", 8192), ("max_concurrency", 4)):
        if row.get(field, supported) != supported:
            raise typer.BadParameter(f"模型目录中的 {field} 必须为 {supported}，容量变更需要重新验收")
    max_input = row.get("max_input_tokens", 8192)
    if type(max_input) is not int or max_input <= 0:
        raise typer.BadParameter("模型目录中的 max_input_tokens 必须是正整数")
    if not row.get("language_model_only", True):
        raise typer.BadParameter("当前服务只支持文本模型配置")
    options = ["--model-name", row["name"], "--model-path", str(Path(row["path"]).resolve()),
               "--dtype", row.get("dtype", "bfloat16"), "--runner", row.get("runner", "aclgraph"),
               "--max-input-tokens", str(max_input),
               "--graph-mode", row.get("graph_mode", "FULL_DECODE_ONLY"),
               "--batch-tokens", str(row.get("batch_tokens", 2048)),
               "--kv-cache-gib", str(row["kv_cache_gib"]),
               "--speculative-tokens", str(row.get("speculative_tokens", 0)),
               "--prefix-caching" if row.get("prefix_caching", True) else "--no-prefix-caching",
               "--async-scheduling" if row.get("async_scheduling", True) else "--no-async-scheduling",
               "--npugraph-ex" if row.get("npugraph_ex", True) else "--no-npugraph-ex"]
    if row.get("overlap_shared_expert", False):
        options.append("--overlap-shared-expert")
    if "model_adaptation" in row:
        options += ["--model-adaptation", row["model_adaptation"]]
    os.execv(sys.executable, [sys.executable, "-m", "qwen_llm.commands.app", "serve", *options, *ctx.args])


@app.command()
def doctor(
    model_path: Path = typer.Option(Path("model-bin/Qwen3-14B-W8A8-Ascend"), envvar="QWEN_MODEL_PATH"),
    probe_runtime: bool = typer.Option(False, envvar="QWEN_PROBE_RUNTIME"),
) -> None:
    """检查安装与量化兼容性；不加载模型或占用 NPU。"""
    from qwen_llm.engine.diagnostics import diagnose
    report = diagnose(model_path, probe_runtime=probe_runtime)
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ready"]:
        raise typer.Exit(1)


@app.command()
def serve(
    model_path: Path = typer.Option(Path("model-bin/Qwen3-14B-W8A8-Ascend"), envvar="QWEN_MODEL_PATH"),
    model_name: str = typer.Option("qwen3-14b", envvar="QWEN_MODEL_NAME"),
    dtype: ModelDType = typer.Option(ModelDType.BFLOAT16, envvar="QWEN_DTYPE"),
    host: str = typer.Option("127.0.0.1", envvar="QWEN_HOST"),
    port: int = typer.Option(8100, min=1, max=65535, envvar="QWEN_PORT"),
    timeout: float = typer.Option(600, min=1, envvar="QWEN_TIMEOUT"),
    max_input_tokens: int = typer.Option(8192, min=1, envvar="QWEN_MAX_INPUT_TOKENS"),
    batch_tokens: int = typer.Option(2048, min=4, envvar="QWEN_BATCH_TOKENS"),
    runner: Runner = typer.Option(Runner.ACLGRAPH, envvar="QWEN_RUNNER"),
    prefix_caching: bool = typer.Option(True, "--prefix-caching/--no-prefix-caching", envvar="QWEN_PREFIX_CACHING"),
    kv_cache_gib: float = typer.Option(10.25, min=0.125, envvar="QWEN_KV_CACHE_GIB"),
    gpu_memory_utilization: float = typer.Option(0.82, min=0.01, max=1, envvar="QWEN_GPU_MEMORY_UTILIZATION"),
    graph_mode: GraphMode = typer.Option(GraphMode.PIECEWISE, envvar="QWEN_GRAPH_MODE"),
    speculative_tokens: int = typer.Option(0, min=0, max=3, envvar="QWEN_SPECULATIVE_TOKENS"),
    async_scheduling: bool | None = typer.Option(None, "--async-scheduling/--no-async-scheduling", envvar="QWEN_ASYNC_SCHEDULING"),
    npugraph_ex: bool | None = typer.Option(None, "--npugraph-ex/--no-npugraph-ex", envvar="QWEN_NPUGRAPH_EX"),
    overlap_shared_expert: bool = typer.Option(False, envvar="QWEN_OVERLAP_SHARED_EXPERT"),
    model_adaptation: ModelAdaptation = typer.Option(
        ModelAdaptation.NONE, envvar="QWEN_MODEL_ADAPTATION", help="已验收的模型专属执行适配。"),
) -> None:
    """启动单 worker V1 Responses 服务。"""
    import uvicorn
    from qwen_llm.config import ServingConfig
    from qwen_llm.servicer.app import create_app
    config = ServingConfig(model_path=model_path, model_name=model_name, dtype=dtype,
                           max_input_tokens=max_input_tokens,
                           timeout=timeout, batch_tokens=batch_tokens,
                           runner=runner, prefix_caching=prefix_caching,
                           kv_cache_bytes=int(kv_cache_gib * 1024**3),
                           gpu_memory_utilization=gpu_memory_utilization,
                           graph_mode=graph_mode, speculative_tokens=speculative_tokens,
                           async_scheduling=async_scheduling, npugraph_ex=npugraph_ex,
                           overlap_shared_expert=overlap_shared_expert, model_adaptation=model_adaptation)
    uvicorn.run(create_app(config), host=host, port=port, workers=1,
                timeout_graceful_shutdown=int(config.shutdown_grace))


@app.command("serve-nvidia")
def serve_nvidia(
    model_path: Path = typer.Option(..., exists=True, envvar="QWEN_MODEL_PATH"),
    model_name: str = typer.Option("qwen38-nvidia", envvar="QWEN_MODEL_NAME"),
    host: str = typer.Option("0.0.0.0", envvar="QWEN_HOST"),
    port: int = typer.Option(8120, min=1, max=65535, envvar="QWEN_PORT"),
    tensor_parallel_size: int = typer.Option(2, min=1, envvar="QWEN_TP"),
    max_model_len: int = typer.Option(16384, min=2, envvar="QWEN_MAX_MODEL_LEN"),
    max_output_tokens: int = typer.Option(8192, min=1, envvar="QWEN_MAX_OUTPUT_TOKENS"),
    max_concurrency: int = typer.Option(4, min=1, envvar="QWEN_MAX_CONCURRENCY"),
    kv_cache_gib: float | None = typer.Option(None, min=0.125, envvar="QWEN_KV_CACHE_GIB", help="每张卡的 KV 预算；未指定时由实际 profile 决定。"),
    batch_tokens: int = typer.Option(2048, min=16, envvar="QWEN_BATCH_TOKENS"),
    gpu_memory_utilization: float = typer.Option(0.9, min=0.1, max=0.98, envvar="QWEN_GPU_MEMORY_UTILIZATION"),
    cuda_graph: bool = typer.Option(False, envvar="QWEN_CUDA_GRAPH"),
    prefix_caching: bool = typer.Option(False, envvar="QWEN_PREFIX_CACHING"),
    timeout: float = typer.Option(3600, min=1, envvar="QWEN_TIMEOUT"),
) -> None:
    """CUDA vLLM TP 服务；须显式设置 CUDA_VISIBLE_DEVICES。"""
    import uvicorn
    from qwen_llm.engine.nvidia import NvidiaConfig, NvidiaEngine
    from qwen_llm.servicer.app import create_app
    if max_model_len <= max_output_tokens:
        raise typer.BadParameter("总上下文必须大于最大输出长度")
    config = NvidiaConfig(
        model_path=model_path, model_name=model_name,
        max_input_tokens=max_model_len - max_output_tokens, max_output_tokens=max_output_tokens,
        max_concurrency=max_concurrency, batch_tokens=batch_tokens, timeout=timeout,
        kv_cache_bytes=int(kv_cache_gib * 1024**3) if kv_cache_gib is not None else None,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size, runner=Runner.CUDAGRAPH if cuda_graph else Runner.EAGER,
        prefix_caching=prefix_caching,
    )
    uvicorn.run(create_app(config, NvidiaEngine(config), max_body_bytes=16 * 1024**2),
                host=host, port=port, workers=1, timeout_graceful_shutdown=int(config.shutdown_grace))


if __name__ == "__main__":
    app()

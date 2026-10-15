from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from importlib.metadata import version

import typer

from qwen_llm.config import ModelDType, Runner

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
    batch_tokens: int = typer.Option(2048, min=4, envvar="QWEN_BATCH_TOKENS"),
    runner: Runner = typer.Option(Runner.ACLGRAPH, envvar="QWEN_RUNNER"),
    prefix_caching: bool = typer.Option(True, "--prefix-caching/--no-prefix-caching", envvar="QWEN_PREFIX_CACHING"),
    kv_cache_gib: float = typer.Option(10.25, min=10, envvar="QWEN_KV_CACHE_GIB"),
) -> None:
    """启动单 worker V1 Responses 服务。"""
    import uvicorn
    from qwen_llm.config import ServingConfig
    from qwen_llm.servicer.app import create_app
    config = ServingConfig(model_path=model_path, model_name=model_name, dtype=dtype,
                           timeout=timeout, batch_tokens=batch_tokens,
                           runner=runner, prefix_caching=prefix_caching,
                           kv_cache_bytes=int(kv_cache_gib * 1024**3))
    uvicorn.run(create_app(config), host=host, port=port, workers=1,
                timeout_graceful_shutdown=int(config.shutdown_grace))


@app.command()
def smoke(
    model_path: Path = typer.Option(Path("model-bin/Qwen3-14B-W8A8-Ascend"), envvar="QWEN_MODEL_PATH"),
) -> None:
    """直接验证真实模型加载、生成与清理；失败返回非零。"""
    from qwen_llm.config import ServingConfig
    from qwen_llm.engine.vllm import V1Engine
    async def run():
        engine = V1Engine(ServingConfig(model_path=model_path))
        try:
            await engine.start()
            prompt = await engine.prepare([{"role": "user", "content": "你好"}], False)
            async for result in engine.generate(prompt, "smoke", max_tokens=16, temperature=0):
                if result.finish_reason:
                    typer.echo(result.text)
        finally:
            await engine.close()
    try:
        asyncio.run(run())
    except Exception as exc:
        logger.error("真实推理验证失败：%s", exc)
        raise typer.Exit(1) from exc


@app.command()
def benchmark(
    base_url: str = typer.Option("http://127.0.0.1:8100/v1", envvar="QWEN_BASE_URL"),
    model_name: str = typer.Option("qwen3-14b", envvar="QWEN_MODEL_NAME"),
    prompt: str = typer.Option("请介绍一下你自己。", envvar="QWEN_BENCH_PROMPT"),
    output_tokens: int = typer.Option(128, min=1, max=8192, envvar="QWEN_BENCH_OUTPUT_TOKENS"),
    output: Path = typer.Option(Path("outputs/benchmark.json"), envvar="QWEN_BENCH_OUTPUT"),
) -> None:
    """在独占设备上预热后测量 1/2/4/8 并发，各三轮；保存原始记录。"""
    from qwen_llm.benchmark import run_benchmark
    asyncio.run(run_benchmark(base_url, model_name, prompt, output_tokens, output))


@app.command("evaluate")
def evaluate_model(
    base_url: str = typer.Option("http://127.0.0.1:8100/v1", envvar="QWEN_BASE_URL"),
    model_name: str = typer.Option("qwen3-14b", envvar="QWEN_MODEL_NAME"),
    output: Path = typer.Option(Path("outputs/quality.json"), envvar="QWEN_EVAL_OUTPUT"),
    math_dataset: Path | None = typer.Option(None, exists=True, dir_okay=False, envvar="QWEN_EVAL_MATH_DATASET"),
    math_samples: int = typer.Option(32, min=1, max=1319, envvar="QWEN_EVAL_MATH_SAMPLES"),
) -> None:
    """保存40道固定答案题的逐请求结果；不代替完整量化精度评估。"""
    from qwen_llm.evaluation import evaluate
    result = asyncio.run(evaluate(base_url, model_name, output, math_dataset, math_samples))
    typer.echo(f"{result['passed']}/{result['total']} passed; execution errors={result['execution_errors']}")
    if result["execution_errors"] or not result["instance_unchanged"]:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

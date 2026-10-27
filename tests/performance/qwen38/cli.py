from __future__ import annotations
import asyncio
import logging
from pathlib import Path
import typer
from qwen_llm.config import GraphMode, ModelDType, ReasoningEffort, Runner
from .config import ProfileConfig, W4A8LinearBackend

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)
app = typer.Typer(no_args_is_help=True, help="Qwen3.8 独立实验工具；不属于生产入口。")

@app.callback()
def main():
    pass

@app.command("quantize-w4a8")
def quantize_w4a8(
    model_path: Path = typer.Option(..., exists=True, file_okay=False, envvar="QWEN_QUANT_SOURCE"),
    output_dir: Path = typer.Option(..., envvar="QWEN_QUANT_OUTPUT"),
    source_manifest: Path = typer.Option(..., exists=True, dir_okay=False, envvar="QWEN_QUANT_SOURCE_MANIFEST"),
    modelslim_root: Path = typer.Option(..., exists=True, file_okay=False, envvar="QWEN_MODELSLIM_ROOT"),
    tool_dependencies: Path = typer.Option(..., exists=True, file_okay=False, envvar="QWEN_QUANT_DEPENDENCIES"),
    w4_scope: str = typer.Option("mlp", envvar="QWEN_QUANT_W4_SCOPE", help="mlp保留注意力/GDN为W8；all扩大W4范围。"),
    group_size: int = typer.Option(256, min=32, envvar="QWEN_QUANT_GROUP_SIZE"),
    threads: int = typer.Option(16, min=1, envvar="QWEN_QUANT_THREADS"),
):
    """从已校验 BF16 原权重导出候选 W4A8；输出仍须真实能力和速度验证。"""
    from tests.performance.qwen38.quantization import export_w4a8
    export_w4a8(model_path, output_dir, source_manifest, modelslim_root,
                tool_dependencies, w4_scope, group_size, threads)


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
            prompt = await engine.prepare([{"role": "user", "content": "你好"}], ReasoningEffort.NONE)
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
    from tests.performance.qwen38.benchmark import run_benchmark
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
    from tests.performance.qwen38.evaluation import evaluate
    result = asyncio.run(evaluate(base_url, model_name, output, math_dataset, math_samples))
    typer.echo(f"{result['passed']}/{result['total']} passed; execution errors={result['execution_errors']}")
    if result["execution_errors"] or not result["instance_unchanged"]:
        raise typer.Exit(1)


@app.command("profile-report")
def profile_report(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False, envvar="QWEN_PROFILE_INPUT"),
) -> None:
    """离线重建算子归因与多实例汇总；不加载模型或占用 NPU。"""
    from tests.performance.qwen38.analysis import summarize_profiles
    result = summarize_profiles(input_dir.resolve())
    typer.echo(f"{len(result['runs'])} runs: {input_dir / 'summary.md'}")


@app.command("profile")
def profile_model(
    output_dir: Path = typer.Option(..., envvar="QWEN_PROFILE_OUTPUT"),
    model_name: str = typer.Option("qwen3.8-27b-mtp2", envvar="QWEN_PROFILE_MODEL"),
    catalog: Path = typer.Option(Path(__file__).resolve().with_name("catalog.json"),
                                 exists=True, dir_okay=False, envvar="QWEN_MODEL_CATALOG"),
    suite: str = typer.Option("full", envvar="QWEN_PROFILE_SUITE"),
    rounds: int = typer.Option(3, min=1, envvar="QWEN_PROFILE_ROUNDS"),
    port: int = typer.Option(18100, min=1, max=65535, envvar="QWEN_PROFILE_PORT"),
    capture: bool = typer.Option(True, "--capture/--no-capture", envvar="QWEN_PROFILE_CAPTURE"),
    cases: str | None = typer.Option(None, envvar="QWEN_PROFILE_CASES"),
    speculative_tokens: int | None = typer.Option(None, min=0, max=3, envvar="QWEN_PROFILE_MTP"),
    runner: Runner | None = typer.Option(None, envvar="QWEN_PROFILE_RUNNER"),
    w4a8_linear_backend: W4A8LinearBackend | None = typer.Option(None, envvar="QWEN_PROFILE_W4_BACKEND"),
    max_input_tokens: int | None = typer.Option(None, min=1, envvar="QWEN_PROFILE_MAX_INPUT"),
    kv_cache_gib: float | None = typer.Option(None, min=0.125, envvar="QWEN_PROFILE_KV_GIB"),
    batch_tokens: int | None = typer.Option(None, min=4, envvar="QWEN_PROFILE_BATCH_TOKENS"),
    mixed_prefill_tokens: int | None = typer.Option(None, min=1548, max=8192),
    quality_samples: int | None = typer.Option(None, min=0, max=1319),
    quality_reference: Path | None = typer.Option(None, exists=True, dir_okay=False),
    long_validation: bool = typer.Option(False),
    cache_boundary_validation: bool = typer.Option(False),
    prefix_growth_validation: bool = typer.Option(False),
    stability_seconds: int = typer.Option(0, min=0),
    optimizations: str = typer.Option("", envvar="QWEN_EXPERIMENT_OPTIMIZATIONS"),
    record_shapes: bool = typer.Option(False, envvar="QWEN_PROFILE_SHAPES"),
    module_ranges: bool = typer.Option(False, envvar="QWEN_PROFILE_MODULES"),
    aic_metrics: str = typer.Option("PipeUtilization", envvar="QWEN_PROFILE_AIC_METRICS"),
) -> None:
    """独占 NPU，运行独立 HTTP 基准及 Worker 算子采集；输出目录必须不存在。"""
    from .config import ProfileConfig
    from tests.performance.qwen38.runner import ProfileSuite, config_from_catalog, run_profile
    try:
        selected_suite = ProfileSuite(suite)
        config = config_from_catalog(catalog, model_name, speculative_tokens=speculative_tokens,
            runner=runner, w4a8_linear_backend=w4a8_linear_backend, max_input_tokens=max_input_tokens,
            kv_cache_bytes=int(kv_cache_gib * 1024**3) if kv_cache_gib is not None else None,
            batch_tokens=batch_tokens, mixed_prefill_tokens=mixed_prefill_tokens,
            optimizations=tuple(x for x in optimizations.split(",") if x))
        if module_ranges and config.runner is not Runner.EAGER:
            raise ValueError("模块标记要求 --runner eager")
        profiling = ProfileConfig(output_dir / "traces", record_shapes, aic_metrics, module_ranges)
        asyncio.run(run_profile(config, profiling, output_dir, suite=selected_suite, rounds=rounds,
                                port=port, capture=capture, only_cases=cases, quality_samples=quality_samples,
                                quality_reference=quality_reference,
                                long_validation=long_validation, cache_boundary_validation=cache_boundary_validation,
                                prefix_growth_validation=prefix_growth_validation,
                                stability_seconds=stability_seconds))
    except (ValueError, RuntimeError) as exc:
        logger.error("热点诊断失败: %s", exc)
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()

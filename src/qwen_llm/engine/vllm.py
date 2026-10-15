import asyncio
import logging
import json
import os

from qwen_llm.config import GraphMode, Runner, ServingConfig
from qwen_llm.engine.base import Generation
from qwen_llm.engine.diagnostics import diagnose

logger = logging.getLogger(__name__)


class V1Engine:
    def __init__(self, config: ServingConfig):
        self.config = config
        self.llm = None
        self.tokenizer = None
        self.runtime_identity = {}
        self.supports_thinking = True

    async def start(self):
        report = diagnose(self.config.model_path)
        if report["blockers"]:
            raise RuntimeError("; ".join(report["blockers"]))
        self.runtime_identity = {key: report.get(key) for key in ("packages", "cann_root", "quantization", "model_quant_type")}
        os.environ["VLLM_USE_V1"] = "1"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        from transformers import AutoTokenizer
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        modern = report["packages"]["vllm"].split("+")[0] == "0.18.0"
        model_config = json.loads((self.config.model_path / "config.json").read_text())
        hybrid = model_config.get("model_type") in {"qwen3_5", "qwen3_5_moe"}
        modern_options = {"language_model_only": True} if modern else {}
        if modern and self.config.async_scheduling is not None:
            modern_options["async_scheduling"] = self.config.async_scheduling
        if modern and self.config.npugraph_ex is not None:
            modern_options["additional_config"] = {
                "ascend_compilation_config": {"enable_npugraph_ex": self.config.npugraph_ex}}
        if modern and hybrid and self.config.prefix_caching:
            modern_options["mamba_cache_mode"] = "align"
        if self.config.speculative_tokens:
            text_config = model_config.get("text_config", model_config)
            if not modern or not hybrid or not text_config.get("mtp_num_hidden_layers", 0):
                raise ValueError("MTP 仅支持带 MTP 权重的 Qwen3.5 架构和新版 vLLM")
            modern_options["speculative_config"] = {
                "method": "qwen3_5_mtp", "num_speculative_tokens": self.config.speculative_tokens,
                "enforce_eager": True}

        self.tokenizer = await asyncio.to_thread(
            AutoTokenizer.from_pretrained, str(self.config.model_path), local_files_only=True)
        self.supports_thinking = "enable_thinking" in str(self.tokenizer.chat_template or "")
        self.runtime_identity["supports_thinking"] = self.supports_thinking
        graph_options = {}
        if self.config.runner is Runner.ACLGRAPH:
            graph_multiplier = self.config.speculative_tokens + 1
            capture_sizes = [n * graph_multiplier for n in (1, 2, 4) if n <= self.config.max_concurrency]
            if self.config.graph_mode is GraphMode.FULL:
                if not modern or hybrid or self.config.speculative_tokens:
                    raise ValueError("完整预填充图仅开放给新版vLLM的非混合、无MTP模型")
                capture_sizes += [n for n in (64, 128, 256) if n <= self.config.batch_tokens]
            graph_options["compilation_config"] = {
                **({"mode": 3} if modern else {"level": 3, "use_inductor": False}),
                "cudagraph_mode": self.config.graph_mode.value,
                "cudagraph_capture_sizes": capture_sizes,
            }
        args = AsyncEngineArgs(
            model=str(self.config.model_path.resolve()), dtype=self.config.dtype.value,
            quantization=report["quantization"],
            worker_cls="qwen_llm.engine.ascend_worker.BudgetedNPUWorker",
            max_model_len=self.config.max_input_tokens + self.config.max_output_tokens,
            max_num_seqs=self.config.max_concurrency,
            max_num_batched_tokens=self.config.batch_tokens,
            enforce_eager=self.config.runner is Runner.EAGER, enable_chunked_prefill=True,
            enable_prefix_caching=self.config.prefix_caching, tensor_parallel_size=1,
            cpu_offload_gb=0, kv_cache_memory_bytes=self.config.kv_cache_bytes,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            **modern_options,
            **graph_options,
        )
        self.llm = AsyncLLM.from_engine_args(args)
        scheduler = getattr(getattr(self.llm, "vllm_config", None), "scheduler_config", None)
        self.runtime_identity["async_scheduling"] = getattr(scheduler, "async_scheduling", None)
        compilation = getattr(getattr(self.llm, "vllm_config", None), "compilation_config", None)
        if compilation is not None:
            self.runtime_identity["graph_mode"] = str(compilation.cudagraph_mode)
            self.runtime_identity["graph_capture_sizes"] = compilation.cudagraph_capture_sizes
        await self.health()
        async for _ in self.generate([self.tokenizer.eos_token_id], "warmup", max_tokens=1):
            pass
        if modern:
            workers = await self.llm.collective_rpc("runtime_status", timeout=30)
            self.runtime_identity["workers"] = workers
            if any(row["full_context_capacity"] < self.config.max_concurrency for row in workers):
                raise ValueError("实际 KV 缓存不足以容纳配置的四路完整上下文")
            if self.config.runner is Runner.ACLGRAPH and any(row["graphs_captured"] == 0 for row in workers):
                raise RuntimeError("请求 ACL Graph，但真实 worker 未捕获执行图")
        logger.info("V1 ready: %s", self.config)

    async def health(self):
        if self.llm is None:
            raise RuntimeError("V1 engine 未就绪")
        await self.llm.check_health()

    async def close(self):
        if self.llm is not None:
            self.llm.shutdown()
            self.llm = None

    async def prepare(self, messages, thinking):
        return await asyncio.to_thread(
            self.tokenizer.apply_chat_template, messages, tokenize=True,
            add_generation_prompt=True, enable_thinking=thinking)

    async def generate(self, prompt, request_id, **sampling):
        thinking = sampling.pop("thinking", False)
        close_think = self.tokenizer.convert_tokens_to_ids("</think>")
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        async for result in self.llm.generate(
            TokensPrompt(prompt_token_ids=prompt), SamplingParams(**sampling), request_id):
            item = result.outputs[0]
            yield Generation(item.text, tuple(item.token_ids), item.finish_reason,
                             getattr(result, "num_cached_tokens", 0),
                             (item.token_ids.index(close_think) + 1 if close_think in item.token_ids
                              else len(item.token_ids)) if thinking else 0)

    async def abort(self, request_id):
        if self.llm is not None:
            await self.llm.abort(request_id)

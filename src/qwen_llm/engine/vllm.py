import asyncio
import logging
import json
import os

from qwen_llm.config import GraphMode, ModelAdaptation, ReasoningEffort, Runner, ServingConfig
from qwen_llm.engine.base import Generation
from qwen_llm.engine.diagnostics import diagnose
from qwen_llm.engine.tokenizer import tokenizer_path

logger = logging.getLogger(__name__)


class V1Engine:
    def __init__(self, config: ServingConfig):
        self.config = config
        self.llm = None
        self.tokenizer = None
        self.runtime_identity = {}
        self.supports_thinking = True
        self.supports_reasoning_effort = False
        self.supports_tools = False

    async def start(self):
        report = diagnose(self.config.model_path)
        if report["blockers"]:
            raise RuntimeError("; ".join(report["blockers"]))
        self.runtime_identity = {key: report.get(key) for key in ("packages", "cann_root", "quantization", "model_quant_type")}
        worker_cls = "qwen_llm.engine.ascend_worker.BudgetedNPUWorker"
        adaptation_options = {}
        if self.config.model_adaptation is ModelAdaptation.QWEN38_910B_V1:
            from qwen_llm.engine.qwen38 import serving_options
            adaptation_options, identity = serving_options(self.config, report["packages"])
            self.runtime_identity["model_adaptation"] = identity
            worker_cls = "qwen_llm.engine.qwen38_worker.Qwen38NPUWorker"
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
            modern_options.setdefault("additional_config", {})['ascend_compilation_config'] = {
                "enable_npugraph_ex": self.config.npugraph_ex}
        if self.config.overlap_shared_expert:
            if not modern or model_config.get("model_type") != "qwen3_5_moe":
                raise ValueError("共享专家流重叠仅用于新版 Qwen3.5 MoE 架构")
            modern_options.setdefault("additional_config", {})["multistream_overlap_shared_expert"] = True
        if modern and hybrid and self.config.prefix_caching:
            modern_options["mamba_cache_mode"] = "align"
        if self.config.speculative_tokens:
            text_config = model_config.get("text_config", model_config)
            if not modern or not hybrid or not text_config.get("mtp_num_hidden_layers", 0):
                raise ValueError("MTP 仅支持带 MTP 权重的 Qwen3.5 架构和新版 vLLM")
            modern_options["speculative_config"] = {
                "method": "qwen3_5_mtp", "num_speculative_tokens": self.config.speculative_tokens,
                "enforce_eager": True}
        if adaptation_options:
            modern_options.setdefault("additional_config", {}).update(adaptation_options.pop("additional_config"))
            modern_options.update(adaptation_options)

        tokenizer_directory = await asyncio.to_thread(tokenizer_path, self.config.model_path)
        self.runtime_identity["tokenizer_path"] = str(tokenizer_directory)
        self.tokenizer = await asyncio.to_thread(
            AutoTokenizer.from_pretrained, str(tokenizer_directory), local_files_only=True)
        template = self.tokenizer.get_chat_template()
        self.supports_thinking = "enable_thinking" in template
        self.supports_reasoning_effort = "reasoning_effort" in template
        self.supports_tools = all(marker in template for marker in ('<tool_call>', '<function=', '<parameter=', 'message.tool_calls', '<tool_response>'))
        self.runtime_identity["supports_thinking"] = self.supports_thinking
        self.runtime_identity["supports_reasoning_effort"] = self.supports_reasoning_effort
        self.runtime_identity["supports_tools"] = self.supports_tools
        self.runtime_identity["reasoning_efforts"] = (
            [effort.value for effort in ReasoningEffort] if self.supports_reasoning_effort else [])
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
            tokenizer=str(tokenizer_directory),
            quantization=report["quantization"],
            worker_cls=worker_cls,
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
            adaptation = self.runtime_identity.get("model_adaptation")
            if adaptation and (not workers or any(
                row.get("model_adaptation", {}).get(key) != adaptation[key]
                for row in workers for key in ("name", "recipe", "manifest_sha256")
            )):
                raise RuntimeError("真实Worker没有启用匹配的模型专项适配")
        logger.info("V1 ready: %s", self.config)

    async def health(self):
        if self.llm is None:
            raise RuntimeError("V1 engine 未就绪")
        await self.llm.check_health()

    async def close(self):
        if self.llm is not None:
            self.llm.shutdown()
            self.llm = None

    async def prepare(self, messages: list[dict], effort: ReasoningEffort, *, tools: list[dict] | None = None) -> list[int]:
        effort = ReasoningEffort(effort)
        kwargs = {"enable_thinking": effort is not ReasoningEffort.NONE}
        if self.supports_reasoning_effort and effort is not ReasoningEffort.NONE:
            kwargs["reasoning_effort"] = effort.value
        if tools:
            if not self.supports_tools:
                raise ValueError("当前模型模板不支持工具调用")
            kwargs["tools"] = tools
        return await asyncio.to_thread(
            self.tokenizer.apply_chat_template, messages, tokenize=True,
            add_generation_prompt=True, **kwargs)

    async def generate(self, prompt, request_id, **sampling):
        thinking = sampling.pop("thinking", False)
        schema = sampling.pop("json_schema", None)
        json_object = sampling.pop("json_object", False)
        close_think = self.tokenizer.convert_tokens_to_ids("</think>")
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        if schema is not None or json_object:
            from vllm.sampling_params import StructuredOutputsParams
            sampling["structured_outputs"] = StructuredOutputsParams(
                json=schema, json_object=True if json_object else None)
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

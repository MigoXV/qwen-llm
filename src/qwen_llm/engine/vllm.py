import asyncio
import logging
import os

from qwen_llm.config import Runner, ServingConfig
from qwen_llm.engine.base import Generation
from qwen_llm.engine.diagnostics import diagnose

logger = logging.getLogger(__name__)


class V1Engine:
    def __init__(self, config: ServingConfig):
        self.config = config
        self.llm = None
        self.tokenizer = None

    async def start(self):
        report = diagnose(self.config.model_path)
        if report["blockers"]:
            raise RuntimeError("; ".join(report["blockers"]))
        os.environ["VLLM_USE_V1"] = "1"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        from transformers import AutoTokenizer
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        self.tokenizer = await asyncio.to_thread(
            AutoTokenizer.from_pretrained, str(self.config.model_path), local_files_only=True)
        graph_options = {}
        if self.config.runner is Runner.ACLGRAPH:
            graph_options["compilation_config"] = {
                "level": 3, "cudagraph_mode": "PIECEWISE", "use_inductor": False,
                "cudagraph_capture_sizes": [n for n in (1, 2, 4) if n <= self.config.max_concurrency],
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
            **graph_options,
        )
        self.llm = AsyncLLM.from_engine_args(args)
        await self.health()
        async for _ in self.generate([self.tokenizer.eos_token_id], "warmup", max_tokens=1):
            pass
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

"""共享 vLLM 请求与生命周期；平台启动由具体 Engine 负责。"""
import asyncio
from collections.abc import Mapping

from qwen_llm.config import InferenceConfig, ReasoningEffort
from qwen_llm.engine.base import Generation


class AsyncGenerationEngine:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.llm = None
        self.tokenizer = None
        self.runtime_identity = {}
        self.supports_thinking = True
        self.supports_reasoning_effort = False
        self.supports_tools = False

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
        encoded = await asyncio.to_thread(
            self.tokenizer.apply_chat_template, messages, tokenize=True,
            add_generation_prompt=True, **kwargs)
        return encoded['input_ids'] if isinstance(encoded, Mapping) else encoded

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

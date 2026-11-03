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
        self.image_placeholder = None
        self.image_pad_token_id = None

    async def health(self):
        if self.llm is None:
            raise RuntimeError("V1 engine 未就绪")
        await self.llm.check_health()

    async def close(self):
        if self.llm is not None:
            self.llm.shutdown()
            self.llm = None

    async def prepare(self, messages: list[dict], effort: ReasoningEffort, *, tools: list[dict] | None = None) -> list[int] | dict:
        effort = ReasoningEffort(effort)
        kwargs = {"enable_thinking": effort is not ReasoningEffort.NONE}
        if self.supports_reasoning_effort and effort is not ReasoningEffort.NONE:
            kwargs["reasoning_effort"] = effort.value
        if tools:
            if not self.supports_tools:
                raise ValueError("当前模型模板不支持工具调用")
            kwargs["tools"] = tools
        images = []
        rendered = []
        for message in messages:
            content = message.get('content')
            if isinstance(content, list):
                parts = []
                for part in content:
                    if part.get('type') == 'image':
                        if self.image_placeholder is None:
                            raise ValueError("当前引擎未启用视觉输入，消息不能包含图像 part")
                        images.append(part['image'])
                        parts.append(self.image_placeholder)
                    else:
                        parts.append(part['text'])
                rendered.append({**message, 'content': ''.join(parts)})
            else:
                rendered.append(message)
        encoded = await asyncio.to_thread(
            self.tokenizer.apply_chat_template, rendered, tokenize=True,
            add_generation_prompt=True, **kwargs)
        ids = encoded['input_ids'] if isinstance(encoded, Mapping) else encoded
        if not images:
            return ids
        ids = list(ids)
        if self.image_pad_token_id is None or ids.count(self.image_pad_token_id) != len(images):
            raise ValueError('chat 模板没有为每张图像保留 image pad 占位符')
        return {'prompt_token_ids': ids, 'multi_modal_data': {'image': images}}

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
        prompt_arg = TokensPrompt(**prompt) if isinstance(prompt, dict) else TokensPrompt(prompt_token_ids=prompt)
        async for result in self.llm.generate(
            prompt_arg, SamplingParams(**sampling), request_id):
            item = result.outputs[0]
            yield Generation(item.text, tuple(item.token_ids), item.finish_reason,
                             getattr(result, "num_cached_tokens", 0),
                             (item.token_ids.index(close_think) + 1 if close_think in item.token_ids
                              else len(item.token_ids)) if thinking else 0)

    async def abort(self, request_id):
        if self.llm is not None:
            await self.llm.abort(request_id)

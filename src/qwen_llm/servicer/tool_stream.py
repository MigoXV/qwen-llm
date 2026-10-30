"""NVIDIA vLLM 原生 Qwen 增量解析的窄适配，不加载模型或执行工具。"""
from qwen_llm.servicer.chat import ToolOutputError, tool_blocks, validate_tool_calls
from qwen_llm.inferencers.text import split_thinking


def supports_tool_stream(engine):
    # NVIDIA 环境固定 vLLM 0.32；不把它的 parser API 引入 Ascend 环境。
    return getattr(engine, 'runtime_identity', {}).get('platform') == 'cuda'


class QwenToolStream:
    def __init__(self, tokenizer, policy, thinking):
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        from vllm.parser.qwen3 import Qwen3Parser

        self.policy, self.thinking = policy, thinking
        self.request = ChatCompletionRequest(model=policy.model, messages=policy.messages,
            tools=policy.tools, tool_choice='auto', include_reasoning=True)
        self.parser = Qwen3Parser(tokenizer, tools=self.request.tools,
                                 chat_template_kwargs={'enable_thinking': thinking})
        self.text = ''
        self.token_count = 0

    def feed(self, generation):
        if not generation.text.startswith(self.text) or len(generation.token_ids) < self.token_count:
            raise RuntimeError('引擎必须返回累计文本和 token IDs')
        result = self.parser.parse_delta(generation.text[len(self.text):],
            list(generation.token_ids[self.token_count:]), self.request, finished=False)
        self.text, self.token_count = generation.text, len(generation.token_ids)
        return result

    def finish(self):
        # 不允许原生 parser 的宽容补全把缺失结束标记变成可执行调用。
        answer = split_thinking(self.text, self.thinking, final=True)[1]
        _, self.blocks = tool_blocks(answer)
        return self.parser.finish_streaming()

    def validate(self, items):
        calls = [{'id': item['call_id'], 'function': {
            'name': item['name'], 'arguments': item['arguments']}} for item in items]
        if [name for name, _ in self.blocks] != [c['function']['name'] for c in calls]:
            raise ToolOutputError('流式工具调用与完整输出不一致')
        validate_tool_calls(calls, self.policy)

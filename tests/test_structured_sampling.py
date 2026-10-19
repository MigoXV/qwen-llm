import sys
from types import SimpleNamespace
import pytest
from qwen_llm.config import ServingConfig
from qwen_llm.engine.vllm import V1Engine


@pytest.mark.parametrize('extra', [{}, {'json_schema': {'type': 'object'}}, {'json_object': True}])
def test_structured_format_reaches_sampler_and_plain_calls_are_unchanged(monkeypatch, extra):
    observed = {}

    def params(**sampling):
        observed.update(sampling)
        return sampling

    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(SamplingParams=params))
    monkeypatch.setitem(sys.modules, 'vllm.inputs', SimpleNamespace(TokensPrompt=lambda **values: values))
    monkeypatch.setitem(sys.modules, 'vllm.sampling_params',
                        SimpleNamespace(StructuredOutputsParams=lambda **values: values))

    class Backend:
        async def generate(self, prompt, sampling, request_id):
            yield SimpleNamespace(outputs=[SimpleNamespace(text='{}', token_ids=[1], finish_reason='stop')],
                                  num_cached_tokens=0)

    async def run():
        engine = V1Engine(ServingConfig())
        engine.llm = Backend()
        engine.tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda _: 2)
        results = [r async for r in engine.generate([1], 'test', max_tokens=16, thinking=False, **extra)]
        assert results[0].text == '{}'

    import asyncio
    asyncio.run(run())
    assert 'json_schema' not in observed and 'json_object' not in observed and 'thinking' not in observed
    if extra:
        assert observed['structured_outputs']['json'] == extra.get('json_schema')
        assert observed['structured_outputs']['json_object'] == (True if extra.get('json_object') else None)
    else:
        assert observed == {'max_tokens': 16}

from openai import OpenAI
import pytest

from tests.test_network import live_service


def test_official_sdk_text_and_stream(live_service):
    url, _, _ = live_service
    with OpenAI(api_key='local-test', base_url=url+'/v1', _strict_response_validation=True) as client:
        response = client.responses.create(model='qwen3-14b', input='hi', max_output_tokens=4)
        assert response.output_text == '你好'
        assert response.usage.input_tokens == 4
        assert response.store is False
        events = list(client.responses.create(model='qwen3-14b', input='hi', stream=True, max_output_tokens=4))
        assert ''.join(event.delta for event in events if event.type=='response.output_text.delta')=='你好'
        assert events[-1].type=='response.completed'
        assert events[-1].response.output_text=='你好'


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('effort', ['none', 'low', 'medium', 'xhigh', 'high'])
def test_sdk_reasoning_efforts(live_service, effort, stream):
    url, _, _ = live_service
    with OpenAI(api_key='local-test', base_url=url+'/v1', _strict_response_validation=True) as client:
        result = client.responses.create(model='qwen3-14b', input='hi',
                                         reasoning={'effort': effort}, stream=stream)
        response = list(result)[-1].response if stream else result
        assert response.status == 'completed'
        assert response.reasoning.effort == ('xhigh' if effort == 'high' else effort)

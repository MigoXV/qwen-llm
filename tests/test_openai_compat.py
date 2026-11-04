"""OpenAI 兼容字段（store / service_tier）请求校验。"""
from pydantic import ValidationError
import pytest

from qwen_llm.servicer.chat import ChatRequest


def _req(**extra):
    return ChatRequest(model='m', messages=[{'role': 'user', 'content': '你好'}], **extra)


def test_store_false_and_service_tier_accepted():
    req = _req(store=False, service_tier='default')
    assert req.store is False


def test_service_tier_auto_accepted():
    assert _req(service_tier='auto').service_tier == 'auto'


def test_store_true_rejected():
    with pytest.raises(ValidationError):
        _req(store=True)


def test_unknown_field_still_forbidden():
    with pytest.raises(ValidationError):
        _req(nonsense_field=1)

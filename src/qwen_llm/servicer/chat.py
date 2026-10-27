"""Chat Completions 协议适配和 Qwen 工具调用解析。"""
from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, Literal

from jsonschema import Draft202012Validator, SchemaError, ValidationError
from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from qwen_llm.servicer.schema import ResponseRequest


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: str
    messages: list[dict[str, Any]] = Field(min_length=1)
    stream: bool = False
    stream_options: dict[str, StrictBool] | None = None
    temperature: float = Field(default=0, ge=0, le=2)
    top_p: float = Field(default=0.95, gt=0, le=1)
    max_tokens: int | None = Field(default=None, gt=0, le=8192)
    max_completion_tokens: int | None = Field(default=None, gt=0, le=8192)
    seed: int | None = None
    reasoning_effort: Literal['none', 'low', 'medium', 'high', 'xhigh'] | None = None
    enable_thinking: StrictBool | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    parallel_tool_calls: StrictBool = True
    n: Literal[1] = 1

    @model_validator(mode='after')
    def validate_request(self):
        if self.stream_options is not None and (not self.stream or set(self.stream_options) - {'include_usage'}):
            raise ValueError('stream_options 仅支持流式请求的 include_usage')
        names = set()
        for tool in self.tools or []:
            if set(tool) != {'type', 'function'} or tool.get('type') != 'function' or not isinstance(tool.get('function'), dict):
                raise ValueError('仅支持 function 工具')
            fn = tool['function']
            if set(fn) - {'name', 'description', 'parameters', 'strict'} or not isinstance(fn.get('name'), str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', fn['name']):
                raise ValueError('无效的 function 定义')
            if fn['name'] in names:
                raise ValueError('工具名称必须唯一')
            names.add(fn['name'])
            if 'parameters' in fn and not isinstance(fn['parameters'], dict):
                raise ValueError('工具参数 schema 必须是对象')
            try:
                Draft202012Validator.check_schema(fn.get('parameters', {'type': 'object'}))
            except SchemaError as exc:
                raise ValueError('无效的工具参数 schema') from exc
        choice = self.choice()
        if choice != 'none' and not names:
            raise ValueError('tool_choice 需要提供工具')
        if choice not in ('auto', 'none', 'required') and choice not in names:
            raise ValueError('指定的工具不存在')
        if choice != 'none' and self.response_format and self.response_format.get('type') != 'text':
            raise ValueError('工具调用和结构化输出不能同时使用')
        self.normalized_messages()
        return self

    def choice(self):
        value = self.tool_choice
        if value is None:
            return 'auto' if self.tools else 'none'
        if isinstance(value, str) and value in ('auto', 'none', 'required'):
            return value
        if isinstance(value, dict) and set(value) == {'type', 'function'} and value['type'] == 'function' and isinstance(value['function'], dict) and set(value['function']) == {'name'} and isinstance(value['function']['name'], str):
            return value['function']['name']
        raise ValueError('无效的 tool_choice')

    def normalized_messages(self):
        result = []
        outstanding = set()
        for msg in self.messages:
            role = msg.get('role')
            if role not in ('system', 'developer', 'user', 'assistant', 'tool'):
                raise ValueError('不支持的 message role')
            allowed = {'role', 'content'} | ({'tool_calls', 'reasoning_content'} if role == 'assistant' else set()) | ({'tool_call_id'} if role == 'tool' else set())
            if set(msg) - allowed:
                raise ValueError('不支持的 message 字段')
            content = msg.get('content')
            if isinstance(content, list):
                if any(not isinstance(part, dict) or set(part) != {'type', 'text'} or part.get('type') != 'text' or not isinstance(part.get('text'), str) for part in content):
                    raise ValueError('仅支持文本 content parts')
                content = ''.join(part['text'] for part in content)
            calls = msg.get('tool_calls')
            if content is None and role == 'assistant' and calls:
                content = ''
            if not isinstance(content, str):
                raise ValueError('message content 必须为文本')
            if role != 'tool' and outstanding:
                raise ValueError('工具调用必须先提供全部工具结果')
            item = {'role': 'system' if role == 'developer' else role, 'content': content}
            if 'reasoning_content' in msg:
                if not isinstance(msg['reasoning_content'], str):
                    raise ValueError('reasoning_content 必须为文本')
                item['reasoning_content'] = msg['reasoning_content']
            if role == 'tool':
                call_id = msg.get('tool_call_id')
                if call_id not in outstanding:
                    raise ValueError('tool_call_id 不匹配')
                outstanding.remove(call_id)
            if calls is not None:
                if not isinstance(calls, list) or not calls:
                    raise ValueError('tool_calls 必须是非空列表')
                item['tool_calls'] = []
                for call in calls:
                    if not isinstance(call, dict) or set(call) != {'id', 'type', 'function'} or call.get('type') != 'function' or not isinstance(call.get('id'), str) or not call['id'] or call['id'] in outstanding:
                        raise ValueError('无效的 tool call')
                    fn = call.get('function')
                    if not isinstance(fn, dict) or set(fn) != {'name', 'arguments'} or not isinstance(fn.get('name'), str) or not isinstance(fn.get('arguments'), str):
                        raise ValueError('无效的 tool function')
                    try:
                        args = json.loads(fn['arguments'])
                        if not isinstance(args, dict):
                            raise ValueError()
                    except (ValueError, TypeError) as exc:
                        raise ValueError('工具参数必须是 JSON 对象') from exc
                    outstanding.add(call['id'])
                    item['tool_calls'].append({'function': {'name': fn['name'], 'arguments': args}})
            result.append(item)
        if outstanding:
            raise ValueError('工具调用缺少结果')
        return result

    def prepare(self):
        fmt = self.response_format or {'type': 'text'}
        if fmt.get('type') == 'json_schema':
            if set(fmt) != {'type', 'json_schema'} or not isinstance(fmt['json_schema'], dict):
                raise ValueError('无效的 response_format')
            fmt = {'type': 'json_schema', **fmt['json_schema']}
        kwargs = dict(model=self.model, input='placeholder', stream=self.stream, temperature=self.temperature,
                      top_p=self.top_p, max_output_tokens=self.max_completion_tokens or self.max_tokens or 2048,
                      seed=self.seed, text={'format': fmt})
        if self.reasoning_effort is not None:
            kwargs['reasoning'] = {'effort': self.reasoning_effort}
        if self.enable_thinking is not None:
            kwargs['enable_thinking'] = self.enable_thinking
        if self.chat_template_kwargs is not None:
            kwargs['chat_template_kwargs'] = self.chat_template_kwargs
        response = ResponseRequest(**kwargs)
        messages = self.normalized_messages()
        # ResponseRequest inserts the structured-output instruction in its system message.
        if fmt.get('type') != 'text':
            messages.insert(0, response.messages()[0])
        choice = self.choice()
        if choice == 'required':
            messages.insert(0, {'role': 'system', 'content': 'You must call at least one of the supplied functions.'})
        elif choice not in ('auto', 'none'):
            messages.insert(0, {'role': 'system', 'content': f'You must call the function {choice}.'})
        return PreparedChat(response, messages, self.tools if choice != 'none' else None)


class PreparedChat:
    def __init__(self, response, messages, tools):
        self.response, self._messages, self.tools = response, messages, tools

    def __getattr__(self, name):
        return getattr(self.response, name)

    def messages(self):
        return self._messages


class ToolOutputError(ValueError):
    pass


CALL = re.compile(r'<tool_call>\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*</tool_call>', re.S)
PARAM = re.compile(r'<parameter=([^>\s]+)>\s*\n?(.*?)\n?\s*</parameter>', re.S)


def parse_tools(text, request):
    if request.choice() == 'none':
        return text, []
    matches = list(CALL.finditer(text))
    residual = CALL.sub('', text)
    if '<tool_call' in residual or '</tool_call>' in residual or '<function=' in residual:
        raise ToolOutputError('工具调用格式不完整或无效')
    choice = request.choice()
    if not matches:
        if choice == 'required' or choice not in ('auto', 'none'):
            raise ToolOutputError('模型未生成要求的工具调用')
        return text, []
    if choice == 'none' or (not request.parallel_tool_calls and len(matches) > 1):
        raise ToolOutputError('模型生成了不允许的工具调用')
    tools = {t['function']['name']: t['function'] for t in request.tools or []}
    calls = []
    for match in matches:
        name, body = match.groups()
        if name not in tools or (choice not in ('auto', 'required') and name != choice):
            raise ToolOutputError('模型调用了未允许的工具')
        params = list(PARAM.finditer(body))
        if PARAM.sub('', body).strip():
            raise ToolOutputError('工具参数格式无效')
        schema = tools[name].get('parameters', {'type': 'object'})
        properties = schema.get('properties', {})
        args = {}
        for param in params:
            key, value = param.groups()
            if key in args:
                raise ToolOutputError('工具参数重复')
            value = value.strip()
            typ = properties.get(key, {}).get('type')
            if typ == 'string':
                args[key] = value
            else:
                try:
                    args[key] = json.loads(value)
                except ValueError:
                    # Union/anyOf schemas may permit a plain string as well as JSON values.
                    args[key] = value
        try:
            Draft202012Validator(schema).validate(args)
        except ValidationError as exc:
            raise ToolOutputError('工具参数不符合 schema') from exc
        calls.append({'id': 'call_' + uuid.uuid4().hex, 'type': 'function',
                      'function': {'name': name, 'arguments': json.dumps(args, ensure_ascii=False)}})
    return residual.strip(), calls


def usage_for(response):
    usage = response.get('usage') or {}
    return {'prompt_tokens': usage.get('input_tokens', 0), 'completion_tokens': usage.get('output_tokens', 0),
            'total_tokens': usage.get('total_tokens', 0)}


def project(response, request, chat_id=None, created=None):
    text = ''.join(part.get('text', '') for item in response.get('output', []) if item['type'] == 'message'
                   for part in item.get('content', []) if part['type'] == 'output_text')
    reasoning = ''.join(part.get('text', '') for item in response.get('output', []) if item['type'] == 'reasoning'
                        for part in item.get('content', []) if part['type'] == 'reasoning_text')
    if response['status'] == 'incomplete' and request.choice() != 'none':
        text, calls = text.split('<tool_call', 1)[0], []
    else:
        text, calls = parse_tools(text, request)
    message = {'role': 'assistant', 'content': (text or None) if calls else text}
    if calls:
        message['tool_calls'] = calls
    if reasoning:
        message['reasoning_content'] = reasoning
    finish = 'length' if response['status'] == 'incomplete' else 'tool_calls' if calls else 'stop'
    return {'id': chat_id or 'chatcmpl_' + uuid.uuid4().hex, 'object': 'chat.completion',
            'created': created or int(time.time()), 'model': request.model,
            'choices': [{'index': 0, 'message': message, 'finish_reason': finish}], 'usage': usage_for(response)}

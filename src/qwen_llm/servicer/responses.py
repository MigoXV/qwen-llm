"""Responses 的扁平函数定义转为现有工具协议，复用校验与模板规则。"""
from qwen_llm.servicer.chat import ChatRequest, PreparedChat
from qwen_llm.servicer.schema import ResponseRequest


def prepare_response(body: ResponseRequest) -> tuple[PreparedChat, ChatRequest]:
    tools = []
    for tool in body.tools or []:
        if tool.get('type') != 'function' or set(tool) - {'type', 'name', 'description', 'parameters', 'strict'}:
            raise ValueError('Responses 仅支持 function 工具及 name、description、parameters、strict 字段')
        function = {key: value for key, value in tool.items() if key != 'type'}
        if function.get('parameters') is None:
            function['parameters'] = {'type': 'object'}
        if function.get('strict') is None:
            function['strict'] = False
        tools.append({'type': 'function', 'function': function})
    choice = body.tool_choice
    if isinstance(choice, dict):
        if set(choice) != {'type', 'name'} or choice.get('type') != 'function':
            raise ValueError('Responses 指定工具须为 {type: function, name: 函数名}')
        choice = {'type': 'function', 'function': {'name': choice['name']}}
    policy = ChatRequest(model=body.model, messages=body.messages(), tools=tools or None,
                         tool_choice=choice, parallel_tool_calls=body.parallel_tool_calls,
                         response_format={'type': body.text.format.type})
    prepared = PreparedChat(body, policy.messages_with_tool_choice(),
                            policy.tools if policy.choice() != 'none' else None)
    return prepared, policy

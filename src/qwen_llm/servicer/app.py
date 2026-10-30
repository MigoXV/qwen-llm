from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import time
import uuid
from jsonschema import Draft202012Validator, ValidationError

import anyio

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from qwen_llm.servicer.limits import BodyLimitMiddleware
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from qwen_llm.config import InferenceConfig, Runner, ServingConfig
from qwen_llm.engine.base import Engine
from qwen_llm.engine.vllm import V1Engine
from qwen_llm.inferencers.text import split_thinking
from qwen_llm.servicer.schema import ResponseRequest
from qwen_llm.servicer.chat import ChatRequest, PreparedChat, ToolOutputError, parse_tools, project
from qwen_llm.servicer.responses import prepare_response
from qwen_llm.servicer.tool_stream import QwenToolStream, supports_tool_stream

logger = logging.getLogger(__name__)


class StructuredOutputError(ValueError):
    """格式失败属于本次请求，不使整个模型实例失效。"""


def error(message, status=400, code="invalid_request_error"):
    return JSONResponse({"error": {"message": message, "type": code, "code": code, "param": None}}, status_code=status)


def create_app(config: InferenceConfig | None = None, engine: Engine | None = None, *, max_body_bytes: int = 1024 * 1024) -> FastAPI:
    config = config or ServingConfig()
    engine = engine or V1Engine(config)
    active: dict[str, asyncio.Task] = {}
    instance_id = uuid.uuid4().hex
    registry = CollectorRegistry()
    requests = Counter("responses_requests_total", "Request outcomes", ["outcome"], registry=registry)
    tokens = Counter("responses_tokens_total", "Completed generation tokens", ["kind"], registry=registry)
    inflight = Gauge("responses_inflight", "Active requests", registry=registry)
    latency = Histogram("responses_duration_seconds", "End to end generation duration", registry=registry)

    @asynccontextmanager
    async def lifespan(app):
        app.state.ready = False
        try:
            await engine.start()
            app.state.ready = True
            app.state.failed = False
            yield
        finally:
            app.state.ready = False
            tasks = list(active.values())
            if tasks:
                _, pending = await asyncio.wait(tasks, timeout=config.shutdown_grace)
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
            await engine.close()

    app = FastAPI(title="Qwen V1 Responses", lifespan=lifespan)
    app.state.ready = False
    app.state.failed = False
    app.add_middleware(BodyLimitMiddleware, max_bytes=max_body_bytes)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        requests.labels("invalid").inc()
        return error(str(exc))

    @app.get("/health/live")
    async def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    async def ready():
        try:
            if not app.state.ready:
                raise RuntimeError("not ready")
            await asyncio.wait_for(engine.health(), timeout=2)
        except Exception:
            app.state.ready = False
            return error("引擎未就绪", 503, "engine_unavailable")
        return {"status": "ready", "model": config.model_name, "engine": "v1",
                "instance_id": instance_id, "max_concurrency": config.max_concurrency,
                "max_input_tokens": config.max_input_tokens,
                "max_output_tokens": config.max_output_tokens,
                "kv_cache_bytes": config.kv_cache_bytes, "enforce_eager": config.runner is Runner.EAGER,
                "gpu_memory_utilization": config.gpu_memory_utilization,
                "requested_runner": config.runner.value, "dtype": config.dtype.value, "prefix_caching": config.prefix_caching,
                "batch_tokens": config.batch_tokens, "graph_mode": config.graph_mode.value,
                "speculative_tokens": config.speculative_tokens,
                "requested_async_scheduling": config.async_scheduling,
                "requested_npugraph_ex": getattr(config, 'npugraph_ex', None),
                "overlap_shared_expert": getattr(config, 'overlap_shared_expert', False),
                "requested_model_adaptation": getattr(getattr(config, 'model_adaptation', None), 'value', 'none'),
                "runtime": getattr(engine, "runtime_identity", {})}

    @app.get("/metrics")
    async def metrics():
        extra = engine.metrics_text() if hasattr(engine, 'metrics_text') else b''
        return Response(generate_latest(registry) + extra, media_type="text/plain; version=0.0.4")

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": config.model_name, "object": "model", "created": 0, "owned_by": "local"}]}

    @app.post("/responses")
    @app.post("/v1/responses")
    async def responses(body: ResponseRequest, request: Request):
        try:
            prepared, policy = prepare_response(body)
        except ValueError as exc:
            return error(str(exc))
        return await generate_response(prepared, request, response_tools=policy)

    async def generate_response(body, request, *, response_tools=None):
        if not app.state.ready:
            return error("引擎未就绪", 503, "engine_unavailable")
        if body.model != config.model_name:
            return error("未知模型", 404, "model_not_found")
        if body.max_output_tokens > config.max_output_tokens:
            return error(f"超出服务输出 token 上限 {config.max_output_tokens}")
        if body.reasoning.effort != "none" and not getattr(engine, "supports_thinking", True):
            return error("当前模型只支持非思考模式，请设置reasoning.effort=none", code="unsupported_parameter")
        if isinstance(body, PreparedChat) and (body.tools or any(m.get('tool_calls') or m['role'] == 'tool' for m in body.messages())) and not getattr(engine, 'supports_tools', False):
            return error("当前模型模板不支持工具调用", code="unsupported_parameter")
        if len(active) >= config.max_concurrency:
            requests.labels("overloaded").inc()
            return error(f"{config.max_concurrency} 路活动请求已满", 429, "rate_limit_exceeded")
        rid = "resp_" + uuid.uuid4().hex
        active[rid] = asyncio.current_task()
        inflight.inc()
        started = time.monotonic()
        deadline = started + config.timeout
        released = False

        def release():
            nonlocal released
            if not released:
                released = True
                active.pop(rid, None)
                inflight.dec()
                latency.observe(time.monotonic() - started)

        async def cleanup():
            try:
                await asyncio.wait_for(engine.abort(rid), timeout=5)
            except Exception:
                logger.exception("abort failed: %s", rid)
            finally:
                release()

        thinking = body.reasoning.effort != "none"
        try:
            tools = body.tools if isinstance(body, PreparedChat) else None
            prompt = await asyncio.wait_for(engine.prepare(body.messages(), body.reasoning.effort, **({'tools': tools} if tools else {})), timeout=config.timeout)
            if len(prompt) > config.max_input_tokens:
                release()
                return error(f"模板展开后输入超过 {config.max_input_tokens} tokens", code="context_length_exceeded")
        except asyncio.CancelledError:
            release()
            raise
        except asyncio.TimeoutError:
            release()
            return error("预处理超时", 504, "request_timeout")
        except Exception:
            release()
            logger.exception("prompt preparation failed")
            return error("输入预处理失败", 500, "internal_error")

        async def events():
            active[rid] = asyncio.current_task()
            sequence = 0
            output = []
            response = {"id": rid, "object": "response", "created_at": int(time.time()),
                        "status": "in_progress", "model": config.model_name, "output": output,
                        "error": None, "incomplete_details": None, "store": False,
                        "previous_response_id": None, "usage": None,
                        "instructions": body.instructions, "metadata": {},
                        "parallel_tool_calls": response_tools.parallel_tool_calls if response_tools else False,
                        "tools": [{'type': 'function', **tool['function']} for tool in response_tools.tools or []] if response_tools else [],
                        "tool_choice": (body.tool_choice or ('auto' if body.tools else 'none')) if response_tools else 'none',
                        "prompt_cache_key": body.prompt_cache_key,
                        "temperature": body.temperature, "top_p": body.top_p,
                        "max_output_tokens": body.max_output_tokens,
                        "reasoning": {"effort": body.reasoning.effort, "summary": None},
                        "text": body.text.model_dump(by_alias=True, exclude_none=True), "truncation": "disabled",
                        "user": None, "background": False, "service_tier": "default"}
            items = {}
            texts = {"reasoning": "", "message": ""}
            final = None
            parse_response_tools = response_tools is not None and response_tools.choice() != 'none'
            tool_stream = None
            call_items = {}

            def event(kind, **data):
                nonlocal sequence
                value = {"type": kind, "sequence_number": sequence, **data}
                sequence += 1
                # Snapshot mutable response/items before later updates.
                return json.loads(json.dumps(value, ensure_ascii=False))

            def changes(reasoning, answer):
                result = []
                for kind, text in (("reasoning", reasoning), ("message", answer)):
                    if not text and kind not in items:
                        continue
                    if kind not in items:
                        idx = len(output)
                        item = {"id": ("rs_" if kind == "reasoning" else "msg_") + uuid.uuid4().hex,
                                "type": kind, "status": "in_progress", "content": []}
                        if kind == "message":
                            item["role"] = "assistant"
                        else:
                            item["summary"] = []
                        part = {"type": "reasoning_text" if kind == "reasoning" else "output_text", "text": ""}
                        if kind == "message":
                            part["annotations"] = []
                        output.append(item)
                        items[kind] = (idx, item, part)
                        result.append(event("response.output_item.added", output_index=idx, item=item))
                        item["content"].append(part)
                        result.append(event("response.content_part.added", output_index=idx, item_id=item["id"], content_index=0, part=part))
                    idx, item, part = items[kind]
                    delta = text[len(texts[kind]):]
                    if delta:
                        prefix = "response.reasoning_text" if kind == "reasoning" else "response.output_text"
                        result.append(event(prefix + ".delta", output_index=idx, item_id=item["id"], content_index=0, delta=delta, **({"logprobs": []} if kind == "message" else {})))
                    texts[kind] = text
                    part["text"] = text
                return result

            def parsed_changes(delta):
                if delta is None:
                    return
                yield from changes(texts['reasoning'] + (delta.reasoning or ''),
                                   texts['message'] + (delta.content or ''))
                for call in delta.tool_calls or []:
                    if call.index not in call_items:
                        if not call.id or not call.function or not call.function.name:
                            raise ToolOutputError('工具调用缺少 ID 或名称')
                        idx = len(output)
                        item = {'id': 'fc_' + uuid.uuid4().hex, 'type': 'function_call',
                                'status': 'in_progress', 'call_id': call.id,
                                'name': call.function.name, 'arguments': ''}
                        output.append(item)
                        call_items[call.index] = (idx, item)
                        yield event('response.output_item.added', output_index=idx, item=item)
                    idx, item = call_items[call.index]
                    if call.function and call.function.arguments:
                        item['arguments'] += call.function.arguments
                        yield event('response.function_call_arguments.delta', output_index=idx,
                                    item_id=item['id'], delta=call.function.arguments)

            iterator = None
            pending = None
            try:
                if parse_response_tools and supports_tool_stream(engine):
                    tool_stream = QwenToolStream(engine.tokenizer, response_tools, thinking)
                yield event("response.created", response=response)
                yield event("response.in_progress", response=response)
                iterator = engine.generate(prompt, rid, max_tokens=body.max_output_tokens,
                                           temperature=body.temperature, top_p=body.top_p, thinking=thinking,
                                           **({"json_schema": body.text.format.json_schema} if body.text.format.type == "json_schema" else {}),
                                           **({"json_object": True} if body.text.format.type == "json_object" else {}),
                                           **({'skip_special_tokens': False} if tool_stream else {}),
                                           **({"seed": body.seed} if body.seed is not None else {})).__aiter__()
                while True:
                    if app.state.failed:
                        raise RuntimeError("engine marked failed")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise asyncio.TimeoutError
                    if await request.is_disconnected():
                        raise asyncio.CancelledError
                    if pending is None:
                        pending = asyncio.create_task(iterator.__anext__())
                    done, _ = await asyncio.wait([pending], timeout=min(0.1, remaining))
                    if not done:
                        continue
                    try:
                        final = pending.result()
                    except StopAsyncIteration:
                        break
                    finally:
                        pending = None
                    if tool_stream:
                        for item in parsed_changes(tool_stream.feed(final)):
                            yield item
                    else:
                        reasoning, answer = split_thinking(final.text, thinking)
                        for item in changes(reasoning, '' if parse_response_tools else answer):
                            yield item
                if final is None or final.finish_reason is None:
                    raise RuntimeError("engine stream ended without final result")
                if final.finish_reason != "length" and body.text.format.type != "text":
                    try:
                        value = json.loads(split_thinking(final.text, thinking, final=True)[1])
                        if body.text.format.type == "json_object" and not isinstance(value, dict):
                            raise StructuredOutputError("expected JSON object")
                        if body.text.format.type == "json_schema":
                            Draft202012Validator(body.text.format.json_schema).validate(value)
                    except (ValueError, ValidationError) as exc:
                        raise StructuredOutputError("structured output does not match request") from exc
                reasoning, answer = split_thinking(final.text, thinking, final=True)
                calls = []
                if tool_stream:
                    if final.finish_reason != 'length':
                        for item in parsed_changes(tool_stream.finish()):
                            yield item
                        tool_stream.validate([item for _, item in call_items.values()])
                elif parse_response_tools:
                    if final.finish_reason == 'length':
                        answer = answer.split('<tool_call', 1)[0]
                    else:
                        answer, calls = parse_tools(answer, response_tools)
                if not tool_stream:
                    for item in changes(reasoning, answer):
                        yield item
                for kind, (idx, item, part) in items.items():
                    prefix = "response.reasoning_text" if kind == "reasoning" else "response.output_text"
                    yield event(prefix + ".done", output_index=idx, item_id=item["id"], content_index=0, text=part["text"], **({"logprobs": []} if kind == "message" else {}))
                    yield event("response.content_part.done", output_index=idx, item_id=item["id"], content_index=0, part=part)
                    item["status"] = "completed"
                    yield event("response.output_item.done", output_index=idx, item=item)
                for idx, item in call_items.values():
                    if final.finish_reason == 'length':
                        item['status'] = 'incomplete'
                    else:
                        yield event('response.function_call_arguments.done', output_index=idx,
                                    item_id=item['id'], arguments=item['arguments'])
                        item['status'] = 'completed'
                    yield event('response.output_item.done', output_index=idx, item=item)
                for call in calls:
                    idx = len(output)
                    item = {'id': 'fc_' + uuid.uuid4().hex, 'type': 'function_call',
                            'status': 'in_progress', 'call_id': call['id'],
                            'name': call['function']['name'], 'arguments': ''}
                    output.append(item)
                    yield event('response.output_item.added', output_index=idx, item=item)
                    arguments = call['function']['arguments']
                    yield event('response.function_call_arguments.delta', output_index=idx,
                                item_id=item['id'], delta=arguments)
                    item['arguments'] = arguments
                    yield event('response.function_call_arguments.done', output_index=idx,
                                item_id=item['id'], arguments=arguments)
                    item['status'] = 'completed'
                    yield event('response.output_item.done', output_index=idx, item=item)
                status = "incomplete" if final.finish_reason == "length" else "completed"
                response["status"] = status
                if status == "incomplete":
                    response["incomplete_details"] = {"reason": "max_output_tokens"}
                nout = len(final.token_ids)
                response["usage"] = {"input_tokens": len(prompt), "output_tokens": nout,
                    "total_tokens": len(prompt) + nout,
                    "input_tokens_details": {"cached_tokens": final.cached_tokens, "cache_write_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": final.reasoning_tokens if thinking else 0}}
                tokens.labels("input").inc(len(prompt))
                tokens.labels("output").inc(nout)
                requests.labels(status).inc()
                yield event("response." + status, response=response)
            except asyncio.CancelledError:
                requests.labels("cancelled").inc()
                raise
            except Exception as exc:
                timeout = isinstance(exc, asyncio.TimeoutError)
                invalid_tool = isinstance(exc, ToolOutputError)
                invalid_output = isinstance(exc, StructuredOutputError) or invalid_tool
                if not timeout and not invalid_output:
                    app.state.ready = False
                    app.state.failed = True
                    logger.exception("generation failed: %s", rid)
                code = "request_timeout" if timeout else "invalid_tool_output" if invalid_tool else "structured_output_invalid" if invalid_output else "engine_error"
                requests.labels(code).inc()
                response.update(status="failed", error={"code": code, "message": "请求超时" if timeout else str(exc) if invalid_tool else "结构化输出不符合请求" if invalid_output else "引擎生成失败"})
                yield event("response.failed", response=response)
            finally:
                # Starlette disconnect uses a level-triggered AnyIO cancel scope.
                # Shield the whole cleanup, including pending-token cancellation.
                with anyio.CancelScope(shield=True):
                    try:
                        if pending is not None:
                            pending.cancel()
                            await asyncio.gather(pending, return_exceptions=True)
                        if iterator is not None:
                            await iterator.aclose()
                    finally:
                        await cleanup()

        if body.stream:
            async def sse():
                async for item in events():
                    yield "event: " + item["type"] + "\ndata: " + json.dumps(item, ensure_ascii=False) + "\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        try:
            last = None
            async for item in events():
                last = item
            result = last["response"]
            if result["status"] == "failed":
                code = result['error']['code']
                return JSONResponse(result, status_code=504 if code == 'request_timeout' else 502 if code == 'invalid_tool_output' else 500)
            return JSONResponse(result)
        finally:
            release()

    @app.post('/v1/chat/completions')
    async def chat_completions(body: ChatRequest, request: Request):
        try:
            prepared = body.prepare()
        except ValueError as exc:
            return error(str(exc))
        parsed_tools = bool(prepared.tools) and supports_tool_stream(engine)
        result = await generate_response(prepared, request, response_tools=body if parsed_tools else None)
        if not body.stream:
            if result.status_code != 200:
                return result
            response = json.loads(result.body)
            try:
                return JSONResponse(project(response, body, parsed_tools=parsed_tools))
            except ToolOutputError as exc:
                return error(str(exc), 502, 'invalid_tool_output')
        if not isinstance(result, StreamingResponse):
            return result

        async def chat_sse():
            chat_id = 'chatcmpl_' + uuid.uuid4().hex
            created = int(time.time())
            def chunk(delta, finish=None):
                return {'id': chat_id, 'object': 'chat.completion.chunk', 'created': created,
                        'model': body.model, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
            def data(value):
                return 'data: ' + json.dumps(value, ensure_ascii=False) + '\n\n'
            iterator = result.body_iterator
            tool_indexes = {}
            try:
                started = False
                async for raw in iterator:
                    if not started:
                        started = True
                        yield data(chunk({'role': 'assistant', 'content': ''}))
                    line = next((line[6:] for line in raw.splitlines() if line.startswith('data: ')), None)
                    if line is None:
                        continue
                    event = json.loads(line)
                    kind = event['type']
                    if kind == 'response.reasoning_text.delta':
                        yield data(chunk({'reasoning_content': event['delta']}))
                    elif kind == 'response.output_text.delta' and (not prepared.tools or parsed_tools):
                        yield data(chunk({'content': event['delta']}))
                    elif parsed_tools and kind == 'response.output_item.added' and event['item']['type'] == 'function_call':
                        item = event['item']
                        index = len(tool_indexes)
                        tool_indexes[item['id']] = index
                        yield data(chunk({'tool_calls': [{'index': index, 'id': item['call_id'],
                            'type': 'function', 'function': {'name': item['name'], 'arguments': ''}}]}))
                    elif parsed_tools and kind == 'response.function_call_arguments.delta':
                        yield data(chunk({'tool_calls': [{'index': tool_indexes[event['item_id']],
                            'function': {'arguments': event['delta']}}]}))
                    elif kind in ('response.completed', 'response.incomplete'):
                        try:
                            projected = project(event['response'], body, chat_id, created, parsed_tools=parsed_tools)
                        except ToolOutputError as exc:
                            yield data({'error': {'message': str(exc), 'type': 'invalid_tool_output', 'code': 'invalid_tool_output'}})
                            return
                        message = projected['choices'][0]['message']
                        if prepared.tools and not parsed_tools and message['content']:
                            yield data(chunk({'content': message['content']}))
                        for index, call in enumerate([] if parsed_tools else message.get('tool_calls', [])):
                            yield data(chunk({'tool_calls': [{'index': index, **call}]}))
                        yield data(chunk({}, projected['choices'][0]['finish_reason']))
                        if body.stream_options and body.stream_options.get('include_usage'):
                            yield data({'id': chat_id, 'object': 'chat.completion.chunk', 'created': created,
                                        'model': body.model, 'choices': [], 'usage': projected['usage']})
                        yield 'data: [DONE]\n\n'
                    elif kind == 'response.failed':
                        yield data({'error': event['response']['error']})
                        return
            finally:
                await iterator.aclose()
        return StreamingResponse(chat_sse(), media_type='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    return app

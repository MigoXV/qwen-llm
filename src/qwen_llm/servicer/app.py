from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import time
import uuid

import anyio

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from qwen_llm.servicer.limits import BodyLimitMiddleware
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from qwen_llm.config import Runner, ServingConfig
from qwen_llm.engine.base import Engine
from qwen_llm.engine.vllm import V1Engine
from qwen_llm.inferencers.text import split_thinking
from qwen_llm.servicer.schema import ResponseRequest

logger = logging.getLogger(__name__)


def error(message, status=400, code="invalid_request_error"):
    return JSONResponse({"error": {"message": message, "type": code, "code": code, "param": None}}, status_code=status)


def create_app(config: ServingConfig | None = None, engine: Engine | None = None) -> FastAPI:
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
    app.add_middleware(BodyLimitMiddleware, max_bytes=1024 * 1024)

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
                "requested_npugraph_ex": config.npugraph_ex,
                "runtime": getattr(engine, "runtime_identity", {})}

    @app.get("/metrics")
    async def metrics():
        return Response(generate_latest(registry), media_type="text/plain; version=0.0.4")

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": config.model_name, "object": "model", "created": 0, "owned_by": "local"}]}

    @app.post("/responses")
    @app.post("/v1/responses")
    async def responses(body: ResponseRequest, request: Request):
        if not app.state.ready:
            return error("引擎未就绪", 503, "engine_unavailable")
        if body.model != config.model_name:
            return error("未知模型", 404, "model_not_found")
        if body.max_output_tokens > config.max_output_tokens:
            return error("超出服务输出 token 上限")
        if body.reasoning.effort != "none" and not getattr(engine, "supports_thinking", True):
            return error("当前模型只支持非思考模式，请设置reasoning.effort=none", code="unsupported_parameter")
        if len(active) >= config.max_concurrency:
            requests.labels("overloaded").inc()
            return error("四路活动请求已满", 429, "rate_limit_exceeded")
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
            prompt = await asyncio.wait_for(engine.prepare(body.messages(), thinking), timeout=config.timeout)
            if len(prompt) > config.max_input_tokens:
                release()
                return error("模板展开后输入超过 8192 tokens", code="context_length_exceeded")
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
                        "parallel_tool_calls": False, "tools": [], "tool_choice": "none",
                        "temperature": body.temperature, "top_p": body.top_p,
                        "max_output_tokens": body.max_output_tokens,
                        "reasoning": {"effort": body.reasoning.effort, "summary": None},
                        "text": {"format": {"type": "text"}}, "truncation": "disabled",
                        "user": None, "background": False, "service_tier": "default"}
            items = {}
            texts = {"reasoning": "", "message": ""}
            final = None

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

            iterator = None
            pending = None
            try:
                yield event("response.created", response=response)
                yield event("response.in_progress", response=response)
                iterator = engine.generate(prompt, rid, max_tokens=body.max_output_tokens,
                                           temperature=body.temperature, top_p=body.top_p, thinking=thinking).__aiter__()
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
                    for item in changes(*split_thinking(final.text, thinking)):
                        yield item
                if final is None or final.finish_reason is None:
                    raise RuntimeError("engine stream ended without final result")
                for item in changes(*split_thinking(final.text, thinking, final=True)):
                    yield item
                for kind, (idx, item, part) in items.items():
                    prefix = "response.reasoning_text" if kind == "reasoning" else "response.output_text"
                    yield event(prefix + ".done", output_index=idx, item_id=item["id"], content_index=0, text=part["text"], **({"logprobs": []} if kind == "message" else {}))
                    yield event("response.content_part.done", output_index=idx, item_id=item["id"], content_index=0, part=part)
                    item["status"] = "completed"
                    yield event("response.output_item.done", output_index=idx, item=item)
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
                if not timeout:
                    app.state.ready = False
                    app.state.failed = True
                    logger.exception("generation failed: %s", rid)
                code = "request_timeout" if timeout else "engine_error"
                requests.labels(code).inc()
                response.update(status="failed", error={"code": code, "message": "请求超时" if timeout else "引擎生成失败"})
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
                return JSONResponse(result, status_code=504 if result["error"]["code"] == "request_timeout" else 500)
            return JSONResponse(result)
        finally:
            release()

    return app

# Chat Completions 联调记录

验证时间：2026-10-27。部署模型为 `qwen3.8-27b-optimized-v1`，使用 Ascend 910B、vLLM V1、MTP2 和 ACL Graph。服务从 `qwen-llm.service` 原有配置重启，`/health/ready` 报告就绪及 `runtime.supports_tools=true`。

- `TORCH_DEVICE_BACKEND_AUTOLOAD=0 poetry run pytest -q`：204 项通过，6 项跳过。关闭 PyTorch 后端自动加载是因为通用 Poetry 环境没有完整安装 Ascend 的运行依赖；真实 NPU 验证使用系统服务自己的运行环境。
- OpenAI Python SDK 3.31 严格响应校验：普通与流式 Chat Completions、强制工具调用、工具结果回传、指定函数流式调用以及既有 Responses 均通过。
- 原版 MGW 的 Go 客户端直接请求 `8100/v1`，模型查询与 Chat Completions 成功；将实际运行的 MGW 配置切换为此地址及当前模型别名，重启后通过其 `/llm/models` 和 `/llm/chat` 验证。
- MeetNote 的 `tests/test_responses.py`、`tests/test_fulltext.py`、`tests/test_engine.py` 共 145 项通过；测试时将 `MEETNOTE_TOKENIZER_PATH` 指向实际部署模型的 tokenizer。
- 系统运行中的 MeetNote 使用原始全文 HTTP 请求和历史转写完成摘要、纪要及行动项，HTTP 200，约 52 秒。全文 `keywords` 为 null 是当前 MeetNote 既有实现的行为，不能将其计为本次通过的关键词生成验证。

实际模型的工具调用是生成式行为；服务对必需工具和指定函数的结果做校验，并不保证每次生成均成功。流式工具调用在完整解析后返回。

会议应用上传 180 秒音频的端到端重试通过：约 28 秒完成转写，约 79 秒完成所有产物；生成 7 条转写、摘要、纪要、工作计划、5 个关键词和 1 个段落摘要。日志确认 MGW 使用 `/v1/chat/completions`，MeetNote 使用 `/v1/responses`。报告位于 `/home/meetnote-e2e-20261018/reports/chat-completions-application-e2e-retry-20261027/report.json`。

首次上传因 Audio Harness 配置的 VAD 地址 `127.0.0.1:50022` 无监听而失败。重试期间临时将此地址转发到现有 VAD 的 `172.16.1.233:50051`，并临时启动原本停止的 `qwen-asr-ascend.service`；测试结束后均已恢复，不更改 Audio Harness 的持久配置。因此要在该环境继续运行音频流程，还需要单独修正 VAD 地址并启用 ASR。

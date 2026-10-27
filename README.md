# qwen-llm

基于 vLLM **V1 AsyncLLM + vllm-ascend + torch_npu** 的无状态文本服务，提供 Responses 和 Chat Completions 接口。HTTP 接口由本项目实现，单 API worker、单 Engine Core 和单模型副本。

初期使用的 14B 权重为 [aleoyang/Qwen3-14B-w8a8-MindIE](https://modelscope.cn/models/aleoyang/Qwen3-14B-w8a8-MindIE)，项目路径 `model-bin/Qwen3-14B-W8A8-Ascend`。实际数据位于 `/home/qwen-llm-build/models/Qwen3-14B-W8A8-Ascend`，项目目录通过符号链接访问，以避开根分区空间不足。权重约 20GB，未修改原始量化文件。当前默认使用下述 Qwen3.8-27B MTP2 配置。

当前默认环境已升级为 **CANN 9.1.1、torch/torch-npu 2.9.1、源码 vLLM 0.18.0 + vLLM-Ascend 0.18.0**，实际由 `torch_npu` 在 Ascend910B4-1 上执行。原有 14B 服务与 ASR 的同负载对照均通过 20% 性能退化门槛，旧环境仍保留。版本、构建补丁、原始数据和回退方法见 [环境升级报告](docs/cann911-torch291.md)。历史配置记录见 [初期验证报告](docs/validation.md)。

## 安装与启动

Python 3.10、Poetry。当前解释器为 `/home/qwen-upgrade-20261015/virtualenvs/qwen-llm-INLN9vSh-py3.10/bin/python`。CANN、依赖、模型和缓存均使用 `/home` 数据盘。

```bash
# 本机已安装；重新构建需要数据盘 CANN 9.1.1 toolkit、910B ops 和 NNAL。
bash scripts/install_ascend.sh
# 27B MTP2：64K总上下文、四路完整容量的独占配置显式使用22.5GiB KV。
bash scripts/run_ascend.sh --kv-cache-gib 22.5
# 显式加载同一环境后运行诊断或测试。
source scripts/env_ascend_291.sh
poetry run qwen-llm doctor --model-path /home/model-bin/Qwen/Qwen3.8-27B-W4A8-Ascend --probe-runtime
# 按模型目录选择配置。
bash scripts/run_model.sh qwen3.6-35b-a3b
# 回退原有 14B 环境。
QWEN_RUNTIME=legacy bash scripts/run_ascend.sh
```

四款新增模型的来源、能力测试与测速见 [量化模型报告](docs/quantized-models.md)。模型文件实际放在 `/home/qwen-models/weights`，`model-bin` 只有符号链接。下载命令应显式指定数据盘，下载器支持断点续传并核对固定 revision、大小和 SHA256：

```bash
poetry run qwen-llm download-model OWNER/MODEL \
  --output-dir /home/qwen-models/weights/MODEL
```

Qwen3.8-27B 的原始 BF16 和自制 W4 权重位于 `/home/model-bin/Qwen/`，量化方案、
NPU 兼容处理、能力验收和性能结果见 [27B W4 报告](docs/qwen38-w4a8.md)。
`qwen3.8-27b-mtp2` 现为默认启动配置，`qwen3.8-27b` 保留为显式低显存选择；
原生 INT8 激活实验 `qwen3.8-27b-w4a8` 有已知算术错误，其实现和别名已迁入测试区，供独立对照，不属于生产入口。

Qwen3.8 的模型专项适配、独立新权重及完整验收见 [专项优化报告](docs/qwen38-optimization.md)。
新别名 `qwen3.8-27b-optimized-v1` 使用独立的 `Qwen3.8-27B-W4A8-Ascend-GDNHeadW8-v1` 仓库，
开启尺度缓存、目标/MTP输出头共享和GDN检查点保留；原始权重、原有默认别名及参数保持不变。
该别名固定使用22.5GiB KV、四路57344输入+8192输出和MTP2，并校验模型产物与依赖版本。

```bash
bash scripts/run_model.sh qwen3.8-27b-optimized-v1
```

`bash scripts/run_ascend.sh`、不带模型名的 `bash scripts/run_model.sh` 与
`poetry run qwen-llm serve-model` 均默认选择 `qwen3.8-27b-mtp2`。
`qwen-llm.service` 调用同一启动脚本，下次启动自动采用此配置。
可用 `QWEN_MODEL_PROFILE` 或 `run_model.sh MODEL` 显式选择其他配置。
已有能力成绩、量化类型与公开报告见 [模型能力汇总](docs/model-capabilities.md)。

服务地址为 `http://127.0.0.1:8100`。原有 14B 别名 `qwen3-14b`，其他模型别名及配置见 `configs/models.json`，每次只启动一个 LLM 实例。启动脚本支持追加 CLI 参数。历史 14B 配置使用 FULL 图、捕获 1/2/4/64/128/256、异步调度、NPUGraphEx、前缀缓存、2048-token 分块预填充和 10.25GiB KV；混合 GDN 模型使用 FULL_DECODE_ONLY，并按测试结果选择 MTP。

`dtype=bfloat16` 指未量化层、残差和 KV 的计算类型，不会据此加载整套 BF16 权重。W8A8 线性层使用 INT8 激活；27B 的 W4 层在当前旧插件中使用 BF16 激活和 INT4 权重反量化算子，具体执行路径见对应报告。`deploy/ascend-291/constraints.txt` 固定推理依赖，`poetry.lock` 管理应用依赖。重新安装推理环境请运行完整安装脚本，单独 `poetry install` 可能覆盖运行时约束。

诊断和测试前加载 `scripts/env_ascend_291.sh`；`doctor --probe-runtime` 不加载模型，测试区CLI的 `smoke` 会加载模型，应先关闭现有 LLM。VS Code 选择上述解释器，调试模块为 `qwen_llm.commands.app`，配置参考 `.env.example`；CLI 不自动读取 `.env`。

## Chat Completions 接口

`POST /v1/chat/completions` 支持 OpenAI SDK 的 `chat.completions.create`；模型别名由 `GET /v1/models` 获取。默认温度为 0、最大输出为 2048 token、关闭思考。支持文本多轮历史、`temperature`、`top_p`、`seed`、`max_tokens`、`max_completion_tokens`（同时指定时优先）、`stream` 和 `stream_options.include_usage`。达到输出上限时 `finish_reason` 为 `length`，调用方应自行判断是否使用截断内容。

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8100/v1", api_key="local-npu")
reply = client.chat.completions.create(
    model="qwen3.8-27b-mtp2",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=256,
)
print(reply.choices[0].message.content)
```

流式对话和函数调用示例：

```python
for chunk in client.chat.completions.create(
    model="qwen3.8-27b-mtp2", messages=[{"role": "user", "content": "你好"}], stream=True,
):
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

messages = [{"role": "user", "content": "查询北京天气"}]
reply = client.chat.completions.create(
    model="qwen3.8-27b-mtp2", messages=messages,
    tools=[{"type": "function", "function": {
        "name": "get_weather", "description": "查询城市天气",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    }}], tool_choice="auto",
)
# 根据 reply.choices[0].message.tool_calls 在应用中执行函数，并将结果作为 tool 消息回传。
```

支持 Qwen 原生工具模板的 `tools` 函数调用、`tool_choice`（`auto`、`none`、`required` 或指定函数）、`parallel_tool_calls` 和工具结果历史；工具由调用方执行，再以 `role=tool`、`tool_call_id` 回传。`required` 和指定函数通过提示和结果校验实现：模型不满足要求时返回错误。流式工具调用在完整解析后发送，不暴露模型的工具标记。无工具能力的模板拒绝工具请求，可通过 `/health/ready` 的 `runtime.supports_tools` 查看。

支持 `reasoning_effort`（`none`、`low`、`medium`、`high`、`xhigh`）、`enable_thinking`、`chat_template_kwargs`；思考内容在扩展字段 `reasoning_content` 中返回。支持 `response_format` 的 `text`、`json_object`、`json_schema`，结构化输出和工具调用不能同时启用。图片、文件及未知参数返回 400。

流式生成开始后如果模型或工具格式出错，会发送包含 `error` 的数据事件并终止，不发送 `[DONE]`；调用方应把没有正常结束的流视为失败。Chat Completions 和 Responses 共享并发名额、超时和监控指标。

## Responses 接口

- `POST /v1/responses`，以及别名 `/responses`。
- `GET /v1/models`：单模型别名。
- `GET /health/live`、`GET /health/ready`：存活及引擎就绪。
- `GET /metrics`：活动请求、结果分类、token 数和生成耗时的 Prometheus 指标。

```bash
curl http://127.0.0.1:8100/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-mtp2","input":"你好","max_output_tokens":256}'

curl -N http://127.0.0.1:8100/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-mtp2","input":"解释一下质数","stream":true,"reasoning":{"effort":"medium"},"max_output_tokens":1024}'
```

支持 `model`、`input`、`instructions`、`stream`、`temperature`、`top_p`、`max_output_tokens`、`reasoning.effort` 和 `store:false`。`input` 为字符串或 message 列表，角色支持 user/assistant/system/developer，developer 按 system 处理；content 支持字符串及 input_text/output_text 文本片段。`instructions` 作为首条 system 消息。

多轮对话每次发送完整 message 历史。响应的 `store` 固定为 false，不提供读取、删除、后台任务、`previous_response_id` 或 conversation 接口。不支持 tools、图片、文件或 reasoning 历史项；结构化输出支持情况见下文。未知参数及不支持的内容返回 400，不默默忽略。

默认关闭思考。Qwen3.8-27B 支持原生 `low`、`medium`、`xhigh` 三档，运行时把档位直接传给原始聊天模板；`none` 关闭思考。`high` 保留为 `xhigh` 的兼容别名，响应 `reasoning.effort` 回显规范化后的实际档位。原生能力见 [官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)及[官方模板](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/chat_template.jinja)。

| `reasoning.effort` | Qwen3.8 行为 |
| --- | --- |
| `none`（服务默认） | 关闭思考 |
| `low` | 简短、集中的思考 |
| `medium` | 平衡准确性和速度 |
| `xhigh` | 更充分的分析与检查 |
| `high`（兼容别名） | 使用 `xhigh`，响应回显 `xhigh` |

```python
# 替换 effort 即可逐请求选择档位；SDK 的 stream=True 同样支持。
client.responses.create(
    model="qwen3.8-27b-optimized-v1", input="计算 17 × 23，并检查结果。",
    reasoning={"effort": "low"}, max_output_tokens=8192,
)
```

也支持 `chat_template_kwargs: {"reasoning_effort": "low"}`。仅传 `enable_thinking=true`（顶层或 `chat_template_kwargs` 内）时使用 `xhigh`；明确指定档位时保留该档位。多个入口的档位必须一致（`high` 与 `xhigh` 等价），开关必须与是否思考一致，否则返回 400。`max_output_tokens` 始终限制思考与正文的总输出，不为各档设置硬性思考预算，也不保证每道题的思考长度严格递增。

思考与正文分别返回 reasoning/message 输出项，流式为 `response.reasoning_text.delta` / `response.output_text.delta`；标签不会作为正文输出。usage 的总输出数量来自实际生成 token IDs，思考计数包含思考区间内生成的控制 token。`/health/ready` 的 `runtime.supports_reasoning_effort` 和 `runtime.reasoning_efforts` 标识原生多档能力；其他旧模型继续使用原有的思考开关语义，不声明原生多档能力。

原始权重核实、问题定位、215 项回归测试及 19 项真实 NPU 场景结果见 [思考档位验收记录](docs/qwen38-reasoning.md)。

SSE 包含 `event` 与递增 `sequence_number`；从 created/in_progress 开始，输出 item/part/delta/done，最终用 completed/incomplete/failed 事件结束，**没有 `[DONE]`**。达到输出限制返回 incomplete，原因 max_output_tokens。输入错误为 400、未知模型为 404、过载为 429、未就绪为 503；非流式超时为 504。已经开始的 SSE 通过 failed 事件表达故障。

## 容量与显存

直接使用`serve`命令的通用默认值为每路模板展开后的输入最多 **8192 tokens**，输出最多 **8192 tokens（含思考）**，总上下文16384。模型目录中的 `max_input_tokens` 会实际传入启动参数 `--max-input-tokens`；也可在 `serve` 中用该选项或 `QWEN_MAX_INPUT_TOKENS` 配置。2026-10-18 会议测试将 Qwen3.6 与 Qwen3.8 MTP2 配置统一为 57344 输入加 8192 输出，总上下文 65536，以保留原应用一次提交全文关键词的行为；MeetNote 内部仍使用 32768 预算；Qwen3.6 KV 为 8.25GiB，Qwen3.8 为 12.5GiB。四个活动名额，不额外排队，第五路返回 429；请求体上限 1MiB。超时默认 600 秒，断连和超时会 abort 并释放名额，关闭宽限期 30 秒。

历史 14B 模型四路完整上下文的 BF16 KV 理论下限为：

```text
2(K/V) × 40层 × 8 KV头 × 128维 × 2字节 × 16384 tokens × 4路 = 10GiB
```

`--gpu-memory-utilization` 默认 0.82，是设备内存 profile 的上限比例；实际 KV 分配仍由 `--kv-cache-gib` 固定，不会按比例填满整卡。这样在 ASR 已运行时也可为 LLM 初始化保留余量。

14B 设置 10.25GiB 留少量块余量，实测 67072 个 KV tokens，最大 16384-token 请求并发容量 4.09。此前 27B MTP2 的 16K 配置使用 9.375GiB KV，包含混合 GDN 状态与草稿模型缓存，实测估算完整上下文容量 4.0 路；当时整卡占用约 40.25GiB。32K 配置的实际缓存容量以 ready 返回的 worker 观测及新验收结果为准。旧插件忽略上游 `kv_cache_memory_bytes`，因此项目 NPUWorker 子类在设备内存 profile 后显式应用预算，超出可用内存则报错；KV预算在项目worker子类中实现，源码兼容补丁单独记录于deploy/ascend-291。

原环境 Eager 实测整卡 HBM 33695MiB，包含无模型时约 3404MiB 的设备基础占用；不能把 KV 预算当成整卡占用。四路 8192-token 输入已测试，模型可能自然结束，因此短输出测试不等于四路满 8192-token 输出实测。

14B 已验证配置使用 FULL 图并包含短输入预填充形状；旧环境保留 PIECEWISE 配置。当前 27B MTP2 使用 FULL_DECODE_ONLY、异步调度、NPUGraphEx 和前缀缓存。MLP 预取在旧配置中实测没有收益，默认关闭。前缀缓存收益需分别报告冷、热缓存。模型专用优化分别记录，MLA 和多卡通信优化不适用于当前单卡 Dense 配置。当前仅一个可见逻辑设备 **0**，对应物理 4；`ASCEND_RT_VISIBLE_DEVICES=4` 会使设备不可见。

## 验证与测速

```bash
poetry check
poetry run pytest -q
poetry run python -m tests.performance.qwen38.cli benchmark --base-url http://127.0.0.1:8100/v1 \
  --model-name qwen3.8-27b-mtp2 \
  --prompt '请连续列出至少100个常见汉语成语，每个成语附一句简短解释，不要省略，不要提前结束。' \
  --output-tokens 128 --output outputs/benchmark.json
```

基准连接已就绪的本机真实服务，检查独占 NPU 和服务身份，预热后对 1/2/4/8 并发各测三轮，保存 SSE 首 token 延迟、吞吐、延迟分位数、响应、拒绝及设备盘点。原始数据不会覆盖为只有平均值的汇总。测试中的假引擎仅用于协议、SDK、超时、断连、过载和生命周期验证，不用于性能结论。

实验、压测、量化导出和热点诊断代码集中在 [`tests/performance/qwen38/`](tests/performance/qwen38/README.md)，不属于生产包。独立热点诊断使用测试模块的 `profile` 命令，复用真实 Responses 服务并在模型 Worker 内采集 Ascend 算子。
无 profiler 的测速与短窗口 trace 分开，输出包含逐轮 HTTP/SSE、引擎 token 批次、调度、图命中、MTP 接受率和原始内核数据。
当前 64K 配置的 12.5GiB KV 仅有约 2.24 路完整上下文容量，启动检查会拒绝四路；诊断实例统一显式使用 22.5GiB。
阶段与算子证据见 [Qwen3.8 热点报告](docs/qwen38-hotspots.md)。

```bash
source scripts/env_ascend_291.sh
poetry run python -m tests.performance.qwen38.cli profile --kv-cache-gib 22.5 --suite full \
  --output-dir /home/qwen-models/reports/qwen3.8-27b/NEW_RUN
# 完整单变量对照、模块映射、硬件计数器及历史配置复现：
bash tests/performance/qwen38/profile_qwen38.sh /home/qwen-models/reports/qwen3.8-27b/NEW_MATRIX
# 离线重新归因和汇总，不加载模型：
poetry run python -m tests.performance.qwen38.cli profile-report /home/qwen-models/reports/qwen3.8-27b/NEW_MATRIX
```

输出目录必须不存在；发现其他 NPU 拥有进程即拒绝测试。`--suite comparison` 测短输入与 8K 冷缓存的单路、四路，
`--cases` 可进一步选择场景；`--speculative-tokens`、`--runner`、`--w4a8-linear-backend` 用于显式单变量对照。
模块标记 `--module-ranges --record-shapes` 仅用于 Eager 映射。诊断结束会关闭独立实例并保存设备释放结果。

## ASR 服务

ASR 位于相邻的 `qwen-asr` 项目，启动器默认使用独立的 Torch 2.9.1 Poetry 环境。性能测试时逐个模型独占 NPU；服务共驻与独占测速的资源数据应分开。

```bash
systemctl start qwen-asr-ascend.service
systemctl is-active qwen-asr-ascend.service
```

本机 LLM 的 systemd 单元模板为 `deploy/systemd/qwen-llm.service`，运行默认 27B MTP2 配置。API 的 `/health/ready` 返回成功后才表示模型就绪；`Type=simple` 单元可能在模型预热期间已显示 active。切换模型前先停止该实例，再启动目标配置；性能基准还需停止 ASR，保证独占 NPU。

```bash
# 安装本机服务模板；启动后检查 API 就绪。
cp deploy/systemd/qwen-llm.service /etc/systemd/system/qwen-llm.service
systemctl daemon-reload
systemctl start qwen-llm.service
curl --fail http://127.0.0.1:8100/health/ready
# 切换到目录中的新模型。
systemctl stop qwen-llm.service
bash scripts/run_model.sh qwen3.6-35b-a3b
```

### 摘要引擎兼容参数

Responses API 接受 `chat_template_kwargs: {"enable_thinking": false}`，
作为 `reasoning.effort="none"` 的兼容写法；仅设置为 true 对应 xhigh。
同时显式提供且互相矛盾时返回参数错误。`chat_template_kwargs` 仅允许
`enable_thinking` 布尔字段和 `reasoning_effort` 档位字段。
可选 `seed` 传递给 vLLM 的 SamplingParams，用于调用方指定采样种子。
系统服务补充 Poetry 所需 PATH；默认仍运行 qwen3.8-27b-mtp2 的 NPU 配置。


## 会议摘要引擎请求兼容

`/v1/responses` 支持当前 MeetNote 原有的 `enable_thinking` 和 `text.format` 请求，不改动会议应用与摘要引擎合同。`enable_thinking` 与原有 `chat_template_kwargs.enable_thinking`、`reasoning.effort` 必须一致，冲突返回 400。

`text.format.type=json_schema` 使用 vLLM 原生结构化采样，传入完整 JSON Schema；`json_object` 使用原生 JSON 模式。完成生成后再次验证输出，格式失败返回 `structured_output_invalid`，释放请求容量并保持模型健康。结构化输出目前只接受非思考模式。未指定格式的调用保持原有行为。

请求所带 JSON Schema 同时投影到模型系统消息，保留字段说明及长度等约束。结构化采样负责限制输出形状，模型仍需读取说明才能理解字段含义；服务不另写业务 Prompt，不改写调用方原有指令和输入。

此变更在 hz01 的 vLLM 0.18 / Ascend 环境运行，Responses、采样转接、SDK 及断连相关共 39 项测试通过；Qwen3.8-27B MTP2 已完成真实结构化推理。命令为 `source scripts/env_ascend_291.sh` 后执行 `poetry run pytest -q tests/test_responses.py tests/test_structured_responses.py tests/test_structured_sampling.py tests/test_sdk.py tests/test_disconnect.py`。

### 32K 输入容量验收

模型配置、CLI 透传、实际 serve 配置和 Responses 输入边界的相关测试共 51 项通过（包含此前结构化输出、SDK、断连用例）。命令在 `source scripts/env_ascend_291.sh` 后执行：

```bash
poetry run pytest -q tests/test_model_profile.py tests/test_responses.py tests/test_structured_responses.py tests/test_structured_sampling.py tests/test_sdk.py tests/test_disconnect.py
```

Qwen3.6 32K 模型已实际接受 18,068 token 的真实会议转写输入并完成结构化响应；worker 报告 KV 实分配约 5.74GiB，完整上下文并发容量约 4.68。Qwen3.8 的 32K 实际模型验收随最终同文比较执行，尚未将配置就绪视为模型验收通过。公开 Responses 字段保持原样，默认输入边界仍由配置决定。

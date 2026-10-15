# qwen-llm

基于 vLLM **V1 AsyncLLM + vllm-ascend + torch_npu** 的无状态文本 Responses 服务。HTTP 接口由本项目实现，单 API worker、单 Engine Core 和单模型副本。

已从 ModelScope 下载并通过 SHA256 校验的模型为 [aleoyang/Qwen3-14B-w8a8-MindIE](https://modelscope.cn/models/aleoyang/Qwen3-14B-w8a8-MindIE)，项目路径 `model-bin/Qwen3-14B-W8A8-Ascend`。实际数据位于 `/home/qwen-llm-build/models/Qwen3-14B-W8A8-Ascend`，项目目录通过符号链接访问，以避开根分区空间不足。权重约 20GB，未修改原始量化文件。

当前默认环境已升级为 **CANN 9.1.1、torch/torch-npu 2.9.1、源码 vLLM 0.18.0 + vLLM-Ascend 0.18.0**，实际由 `torch_npu` 在 Ascend910B4-1 上执行。原有 14B 服务与 ASR 的同负载对照均通过 20% 性能退化门槛，旧环境仍保留。版本、构建补丁、原始数据和回退方法见 [环境升级报告](docs/cann911-torch291.md)。历史配置记录见 [初期验证报告](docs/validation.md)。

## 安装与启动

Python 3.10、Poetry。当前解释器为 `/home/qwen-upgrade-20261015/virtualenvs/qwen-llm-INLN9vSh-py3.10/bin/python`。CANN、依赖、模型和缓存均使用 `/home` 数据盘。

```bash
# 本机已安装；重新构建需要数据盘 CANN 9.1.1 toolkit、910B ops 和 NNAL。
bash scripts/install_ascend.sh
# 默认启动原有 14B W8A8 服务。
bash scripts/run_ascend.sh
# 显式加载同一环境后运行诊断或测试。
source scripts/env_ascend_291.sh
poetry run qwen-llm doctor --probe-runtime
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

服务地址为 `http://127.0.0.1:8100`。原有 14B 别名 `qwen3-14b`，其他模型别名及配置见 `configs/models.json`，每次只启动一个 LLM 实例。启动脚本支持追加 CLI 参数。默认 14B 使用 FULL 图、捕获 1/2/4/64/128/256、异步调度、NPUGraphEx、前缀缓存、2048-token 分块预填充和 10.25GiB KV；混合 GDN 模型使用 FULL_DECODE_ONLY，并按测试结果选择 MTP。

`dtype=bfloat16` 指未量化层、残差和 KV 的计算类型；模型线性层使用量化权重及 INT8 激活，不会据此加载整套 BF16 权重。`deploy/ascend-291/constraints.txt` 固定推理依赖，`poetry.lock` 管理应用依赖。重新安装推理环境请运行完整安装脚本，单独 `poetry install` 可能覆盖运行时约束。

诊断和测试前加载 `scripts/env_ascend_291.sh`；`doctor --probe-runtime` 不加载模型，`smoke` 会加载模型，应先关闭现有 LLM。VS Code 选择上述解释器，调试模块为 `qwen_llm.commands.app`，配置参考 `.env.example`；CLI 不自动读取 `.env`。

## Responses 接口

- `POST /v1/responses`，以及别名 `/responses`。
- `GET /v1/models`：单模型别名。
- `GET /health/live`、`GET /health/ready`：存活及引擎就绪。
- `GET /metrics`：活动请求、结果分类、token 数和生成耗时的 Prometheus 指标。

```bash
curl http://127.0.0.1:8100/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-14b","input":"你好","max_output_tokens":256}'

curl -N http://127.0.0.1:8100/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-14b","input":"解释一下质数","stream":true,"reasoning":{"effort":"medium"},"max_output_tokens":1024}'
```

支持 `model`、`input`、`instructions`、`stream`、`temperature`、`top_p`、`max_output_tokens`、`reasoning.effort` 和 `store:false`。`input` 为字符串或 message 列表，角色支持 user/assistant/system/developer，developer 按 system 处理；content 支持字符串及 input_text/output_text 文本片段。`instructions` 作为首条 system 消息。

多轮对话每次发送完整 message 历史。响应的 `store` 固定为 false，不提供读取、删除、后台任务、`previous_response_id` 或 conversation 接口。首版不支持 tools、图片、文件、结构化输出或 reasoning 历史项；未知参数及不支持的内容返回 400，不默默忽略。

默认关闭思考；`reasoning.effort=none` 关闭，low/medium/high 均启用 Qwen3 原生思考，不承诺不同档位具有不同计算深度。思考与正文分别返回 reasoning/message 输出项，流式为 `response.reasoning_text.delta` / `response.output_text.delta`；标签不会作为正文输出。usage 的总输出数量来自实际生成 token IDs，思考计数包含思考区间内生成的控制 token。

SSE 包含 `event` 与递增 `sequence_number`；从 created/in_progress 开始，输出 item/part/delta/done，最终用 completed/incomplete/failed 事件结束，**没有 `[DONE]`**。达到输出限制返回 incomplete，原因 max_output_tokens。输入错误为 400、未知模型为 404、过载为 429、未就绪为 503；非流式超时为 504。已经开始的 SSE 通过 failed 事件表达故障。

## 容量与显存

每路模板展开后的输入最多 **8192 tokens**，输出最多 **8192 tokens（含思考）**，总上下文 16384。四个活动名额，不额外排队，第五路返回 429；请求体上限 1MiB。超时默认 600 秒，断连和超时会 abort 并释放名额，关闭宽限期 30 秒。

四路完整上下文的 BF16 KV 理论下限为：

```text
2(K/V) × 40层 × 8 KV头 × 128维 × 2字节 × 16384 tokens × 4路 = 10GiB
```

`--gpu-memory-utilization` 默认 0.82，是设备内存 profile 的上限比例；实际 KV 分配仍由 `--kv-cache-gib` 固定，不会按比例填满整卡。这样在 ASR 已运行时也可为 LLM 初始化保留余量。

设置 10.25GiB 留少量块余量，实测 67072 个 KV tokens，最大 16384-token 请求并发容量 4.09。旧插件忽略上游 `kv_cache_memory_bytes`，因此项目 NPUWorker 子类在设备内存 profile 后显式应用预算，超出可用内存则报错；不修改厂商源码。

原环境 Eager 实测整卡 HBM 33695MiB，包含无模型时约 3404MiB 的设备基础占用；不能把 KV 预算当成整卡占用。四路 8192-token 输入已测试，模型可能自然结束，因此短输出测试不等于四路满 8192-token 输出实测。

当前 14B 默认 FULL 图并包含短输入预填充形状；旧环境保留 PIECEWISE 配置。MLP 预取在旧配置中实测没有收益，默认关闭。前缀缓存收益需分别报告冷、热缓存。模型专用优化分别记录，MLA 和多卡通信优化不适用于当前单卡 Dense 配置。当前仅一个可见逻辑设备 **0**，对应物理 4；`ASCEND_RT_VISIBLE_DEVICES=4` 会使设备不可见。

## 验证与测速

```bash
poetry check
poetry run pytest -q
poetry run qwen-llm benchmark --base-url http://127.0.0.1:8100/v1 \
  --prompt '请连续列出至少100个常见汉语成语，每个成语附一句简短解释，不要省略，不要提前结束。' \
  --output-tokens 128 --output outputs/benchmark.json
```

基准连接已就绪的本机真实服务，检查独占 NPU 和服务身份，预热后对 1/2/4/8 并发各测三轮，保存 SSE 首 token 延迟、吞吐、延迟分位数、响应、拒绝及设备盘点。原始数据不会覆盖为只有平均值的汇总。测试中的假引擎仅用于协议、SDK、超时、断连、过载和生命周期验证，不用于性能结论。

## ASR 服务

ASR 位于相邻的 `qwen-asr` 项目，启动器默认使用独立的 Torch 2.9.1 Poetry 环境。性能测试时逐个模型独占 NPU；服务共驻与独占测速的资源数据应分开。

```bash
systemctl start qwen-asr-ascend.service
systemctl is-active qwen-asr-ascend.service
```

本机 LLM 的 systemd 单元模板为 `deploy/systemd/qwen-llm.service`，运行默认14B配置。API 的 `/health/ready` 返回成功后才表示模型就绪；`Type=simple` 单元可能在模型预热期间已显示 active。切换模型前先停止该实例，再启动目标配置；性能基准还需停止 ASR，保证独占 NPU。

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

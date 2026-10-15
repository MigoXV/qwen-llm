# qwen-llm

基于 vLLM **V1 AsyncLLM + vllm-ascend + torch_npu** 的无状态文本 Responses 服务。HTTP 接口由本项目实现，单 API worker、单 Engine Core 和单模型副本。

已从 ModelScope 下载并通过 SHA256 校验的模型为 [aleoyang/Qwen3-14B-w8a8-MindIE](https://modelscope.cn/models/aleoyang/Qwen3-14B-w8a8-MindIE)，项目路径 `model-bin/Qwen3-14B-W8A8-Ascend`。实际数据位于 `/home/qwen-llm-build/models/Qwen3-14B-W8A8-Ascend`，项目目录通过符号链接访问，以避开根分区空间不足。权重约 20GB，未修改原始量化文件。

当前模型已在 Ascend910B4-1 上完成真实推理、四路长输入和过载测试。关闭前缀缓存的短输入128-token输出基准：单路45.9tokens/s，四路合计171.0tokens/s，相比eager约三倍；最终配置四路8K输入期间整卡峰值约33.4GiB。性能记录及适用范围见 [验证报告](docs/validation.md)。原 `Qwen3-14B-AWQ-ms` 保留；本机插件不支持其标准 AWQ 格式。

## 安装与启动

Python 3.10、Poetry；本次实际运行环境位于 `/home/qwen-llm-build/virtualenvs/qwen-llm-INLN9vSh-py3.10`。原 `/root/.cache/pypoetry/virtualenvs/...` 是另一套较新 Torch 环境，没有安装本次固定的 vLLM 栈。

```bash
export POETRY_CACHE_DIR=/home/qwen-llm-build/poetry-cache
export POETRY_VIRTUALENVS_PATH=/home/qwen-llm-build/virtualenvs
# 首次安装：源码安装固定版本 vLLM，再安装 Ascend 插件。
bash scripts/install_ascend.sh
# 下载支持断点续传、固定文件 revision，并检查大小和 SHA256。
poetry run qwen-llm download-model aleoyang/Qwen3-14B-w8a8-MindIE \
  --output-dir model-bin/Qwen3-14B-W8A8-Ascend
# 本机启动：加载 CANN、与 Torch ABI 匹配的 ATB 和独立 Poetry 环境。
bash scripts/run_ascend.sh
```

服务地址 `http://127.0.0.1:8100`，模型名 `qwen3-14b`。8000 已有其他服务占用。`run_ascend.sh` 可追加 `--runner eager` / `--runner aclgraph`、`--prefix-caching` / `--no-prefix-caching` 等选项。

运行配置为 BF16 计算、Ascend W8A8 量化、PIECEWISE ACL Graph、前缀缓存、2048-token 分块预填充、四路容量、10.25GiB KV 预算。FP16 在当前 CANN 的 AddRmsNormQuantV2 融合算子上报类型不支持，使用 BF16 保留融合加速。

`scripts/ascend-constraints.txt` 固定历史推理依赖；`poetry.lock` 管理应用依赖。不要在已配置的推理环境中随意重跑 `poetry install`，这会覆盖推理依赖；需要重新安装时运行完整安装脚本恢复约束。脚本不修改系统 CANN、驱动或原 ASR 环境。完整构建、下载和运行日志在 `/home/qwen-llm-build/logs`。

直接运行 doctor、smoke 或 serve 前也需加载 CANN 与 ATB；可以参照 `scripts/run_ascend.sh`。`doctor --probe-runtime` 只探测环境，不加载模型；`smoke` 会加载模型，应先关闭现有实例。VS Code 选择上述实际运行解释器，调试入口为 `qwen_llm.commands.app`，环境变量见 `.env.example`；CLI 不自动读取 `.env`。

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

设置 10.25GiB 留少量块余量，实测 67072 个 KV tokens，最大 16384-token 请求并发容量 4.09。旧插件忽略上游 `kv_cache_memory_bytes`，因此项目 NPUWorker 子类在设备内存 profile 后显式应用预算，超出可用内存则报错；不修改厂商源码。

Eager 实测整卡 HBM 33695MiB，包含无模型时约 3404MiB 的设备基础占用；不能把 KV 预算当成整卡占用。四路 8192-token 输入已测试，模型可能自然结束，因此短输出测试不等于四路满 8192-token 输出实测。

ACL Graph 使用 PIECEWISE、捕获批次 1/2/4，禁用此插件不支持的 Inductor；MLP预取实测没有收益，默认关闭。前缀缓存收益需分别报告冷、热缓存。TorchAirGraph、MLA、MoE、多卡通信优化不适用于当前单卡 Qwen3 Dense 配置。当前仅一个可见逻辑设备 **0**，对应物理 4；`ASCEND_RT_VISIBLE_DEVICES=4` 会使设备不可见。

## 验证与测速

```bash
poetry check
poetry run pytest -q
poetry run qwen-llm benchmark --base-url http://127.0.0.1:8100/v1 \
  --prompt '请连续列出至少100个常见汉语成语，每个成语附一句简短解释，不要省略，不要提前结束。' \
  --output-tokens 128 --output outputs/benchmark.json
```

基准连接已就绪的本机真实服务，检查独占 NPU 和服务身份，预热后对 1/2/4/8 并发各测三轮，保存 SSE 首 token 延迟、吞吐、延迟分位数、响应、拒绝及设备盘点。原始数据不会覆盖为只有平均值的汇总。测试中的假引擎仅用于协议、SDK、超时、断连、过载和生命周期验证，不用于性能结论。

## ASR 恢复

已按授权停止 `qwen-asr-ascend.service`，保留开机启动设置。恢复前关闭本项目模型，避免共享 NPU 干扰。

```bash
systemctl start qwen-asr-ascend.service
systemctl is-active qwen-asr-ascend.service
```

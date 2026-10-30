# GPU 2：24 GiB、两路 FP8 KV 服务

2026-10-29 按用户要求保留运行。监听 `0.0.0.0:8132`，本机地址 `http://127.0.0.1:8132/v1`，服务器地址 `http://192.168.0.223:8132/v1`，模型名 `qwen38-24g`。已通过服务器 IP 同时发送中英文请求验证。

| 参数 | 实际配置 |
|---|---|
| GPU / TP | 物理 GPU 2 / TP1 |
| 模型 | `Qwen3.8-27B-Abliterated-W4A16-AWQ-GPTQ-VOCAB4-NVIDIA` |
| KV cache | FP8 E4M3，固定预算 8.75 GiB |
| 同时请求 | 2 |
| 每路总上下文 | 131072 token（128K） |
| 每路最大输入 / 输出 | 98304 / 32768 token（96K / 32K） |
| 后端 | FLASHINFER，FULL_DECODE_ONLY CUDA Graph |
| batch tokens / 前缀缓存 / MTP | 512 / 开启（GDN align） / 关闭 |

为兼容 Responses 客户端的 `max_output_tokens=32768`，输出上限为 32K，输入上限为 96K，总上下文为 128K。开启前缀缓存后，为容纳额外 GDN 状态，KV 预算从 8.5 调到 8.75 GiB，实际池计算的完整上下文同时驻留容量为 2.02222。3 号卡同配置已实测两路分别完成 96K 输入加完整 32K 输出，零缓存命中、零抢占；2 号卡只做部署后的功能检查，没有重复整轮压力测试。完整数据见 [Responses 与缓存报告](nvidia-responses-cache.md)。更早的 248K 单路检索结果见 [24 GiB 报告](nvidia-24g-hotspots.md)。模型支持中英文，不需要额外语言开关。

监督器每 100 ms 采样 GPU 2 的**总 NVML 显存**，超过 24 GiB 时停止自己的服务；监测异常也停止服务。它不是硬件分区，不能保证采样之间的短时尖峰。3 号卡同配置完整容量测试峰值为 23.7514 GiB；2 号卡本次部署和功能检查约 23.64 GiB，实时值见 `budget-status.json`。早期自动 KV 预算的编译准备曾触到约 24.37 GiB，被监督器终止；此后使用固定预算、已有缓存启动，不把该失败写成预算内成功。

配置文件：[gpu2-24g-c2.json](../deploy/nvidia/gpu2-24g-c2.json)。监督器：[serve_nvidia_budget.py](../scripts/serve_nvidia_budget.py)。它持续运行，没有此前容量实验的两小时退出计时；显存统计只保留计数/峰值，采样日志轮转，不在内存中无限保存所有样本。

当前日志和状态：`outputs/services/gpu2-24g-c2-prefix-20261029/`，实例 ID 为 `f6336647e4ec45b0b8e4dbcf7e25b08a`。其中 `process.json` 保存命令、PID 和源码哈希，`ready.json` 保存实例配置，`budget-status.json` 每约五秒更新，`deployment-smoke.json` 保存通过服务器 IP 完成的 SDK 检查。先前 `gpu2-24g-c2-responses-20261029/` 和 `gpu2-24g-c2-public-20261029/` 实例均已退出，历史日志保留。

Responses 修复已通过 169 项回归检查，3 项依赖本地 Ascend tokenizer 的检查跳过。真实服务使用官方 Python SDK 严格响应校验；NVIDIA 正文和工具参数现在随生成发送，结束时再确认校验结果，修复了 DSH 因一次性发送而显示几千甚至十万 TPS 的问题。工具仍由客户端执行。短输入带 tools 的普通回答实测单路约 69 token/s、两路每路约 66 token/s；16K 缓存命中时单路首字约 0.45–0.49 秒。

前缀缓存已开启，实际命中数见 `usage.input_tokens_details.cached_tokens`，不是回答缓存。`prompt_cache_key` 接收并回显为兼容提示，实际复用按 token 前缀匹配；它不提供会话存储或历史恢复。多轮调用须回传完整 `input` 历史。旧会话已经记下的错误 TPS 不会自动重算，检查修复效果时使用新的请求或会话。

重启时先停止旧实例，再为配置中的 `run_dir` 指定新日志目录，保留现有 `cache`。从项目根目录执行：

```bash
export QWEN_NVIDIA_ENV=runtime QWEN_INSTANCE=gpu2-24g-service
source scripts/env_nvidia.sh
poetry run python -m scripts.serve_nvidia_budget deploy/nvidia/gpu2-24g-c2.json
```

请求示例：

```bash
curl http://192.168.0.223:8132/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-24g","messages":[{"role":"user","content":"用中文介绍一下你自己"}],"max_tokens":256}'
```

纯文本流式调用（需要 `jq`，已实测）：

```bash
curl -sS -N http://192.168.0.223:8132/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-24g","messages":[{"role":"user","content":"你好，请用中文介绍一下你自己。"}],"stream":true,"max_tokens":512}' \
  | sed -u -n '/^data: \[DONE\]/d; s/^data: //p' \
  | jq --unbuffered -jr '.choices[0].delta.content // empty'
echo
```

需要停止当前实例时创建 `outputs/services/gpu2-24g-c2-prefix-20261029/stop.json`，或向监督器发送 SIGTERM；监督器负责停止其 API/引擎进程。回退缓存配置时将 `prefix_caching` 改为 false、`kv_gib` 改回 8.5，并使用新的 `run_dir` 重启；流式修复不依赖缓存开关。未采用的手写算子及其专用实验工具已清理，当前服务使用 vLLM 原生 Marlin。

NVIDIA 普通入口和实验入口现在都默认 `--host 0.0.0.0`，可用 `--host` 或 `QWEN_HOST` 显式覆盖。Ascend 入口未修改。

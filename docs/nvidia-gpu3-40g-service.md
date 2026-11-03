# GPU3 原版 VOCAB4：40 GiB、四路配置

2026-11-03 已切换并就绪，替代此前同端口的 24 GiB 两路实例。

| 参数 | 当前值 |
|---|---|
| 物理 GPU / TP | 3 / 1 |
| 访问地址 / 模型名 | `http://192.168.0.223:8132/v1` / `qwen38-24g`（保留客户端兼容名称） |
| 权重 | `/workspace/opusi/Qwen/Qwen3.8-27B-W4A16-AWQ-GPTQ-VOCAB4-NVIDIA` |
| NVML 总显存监督上限 | 40 GiB，100ms 采样；超额或监控失败关闭本实例 |
| 并发 / 总上下文 | 4 / 131072 token |
| 每请求最大输入 / 输出 | 98304 / 32768 token |
| 固定 KV / dtype | 22 GiB / FP8 E4M3 |
| utilization / batch tokens | 0.95 / 512 |
| 后端 / 图 | FLASHINFER / FULL_DECODE_ONLY CUDA Graph |
| 前缀缓存 / MTP | 开启（GDN align） / 关闭 |

四路功能检查：通过服务器 IP 同时提交四个请求，每个实际输出 512 token，均正常结束，耗时约 8.3 秒，实例未变；显存采样峰值约 39.22 GiB，未超限。这是短请求并发检查，未重新执行四路完整 128K 长输出容量压力测试。

配置：[gpu3-original-vocab4-40g-c4.json](../deploy/nvidia/gpu3-original-vocab4-40g-c4.json)。监督器支持可选 `budget_gib`，省略时仍默认 24 GiB；`port` 保留 8132。

运行证据：`outputs/services/gpu3-original-vocab4-40g-c4-20261103-r2/`，包括 `ready.json`、`process.json`、`budget-status.json`、`four-request-smoke.json` 和轮转显存日志。首次切换尝试因旧实例尚未释放 GPU 而被隔离检查拒绝，未启动模型；待旧实例退出后使用新日志目录重启成功。

重启需先正常停止当前监督器，确认 GPU3 进程退出，并将配置的 `run_dir` 改为全新目录，再执行：

```bash
export QWEN_NVIDIA_ENV=runtime QWEN_INSTANCE=original-vocab4-40g-c4
source scripts/env_nvidia.sh
poetry run python -m scripts.serve_nvidia_budget deploy/nvidia/gpu3-original-vocab4-40g-c4.json
```

模型质量边界沿用 `outputs/original-20261103-gpu2/README.md` 的原始评估；本次只验证配置切换、显存预算与四路功能，没有重新运行完整质量题库。

## 160K 容量复验及当前服务

当前配置已提升到 163840 总上下文：131072 输入、32768 输出，四路并发；端口仍为 8132。四路各 131072 输入加完整 32768 输出实测通过，零命中、零抢占，耗时 1327.64 秒，总吞吐 98.73 TPS，峰值约 39.21 GiB。22 GiB KV 池计算的 160K 驻留容量为 4.135 路，下一档 168K 为 3.957 路，仅为计算边界，未实测失败。

证据：`outputs/services/gpu3-40g-capacity-20261103/result.json`。已恢复非强制输出正常服务，运行目录 `outputs/services/gpu3-original-vocab4-40g-c4-160k-20261103/`。

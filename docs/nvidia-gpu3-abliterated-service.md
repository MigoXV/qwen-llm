# GPU3 Abliterated VOCAB4：256K、两路服务

2026-11-07 按用户要求停止原版 Qwen3.8 160K 四路实例，切换为此前的 Abliterated 量化产物。旧权重保留。

- 地址：`http://192.168.0.223:8132/v1`；监听 `0.0.0.0:8132`。
- 模型调用名：`qwen38-24g`，为兼容旧客户端保留名称；本实例预算实际为40GiB。
- 模型：`/workspace/model-bin/windowsxp811203/Qwen3.8-27B-Abliterated-W4A16-AWQ-GPTQ-VOCAB4-NVIDIA`。
- 物理GPU3、TP1、两路并发，总上下文262144；每请求最大输入229376、最大输出32768。
- FP8 E4M3 KV固定22GiB，utilization 0.95，batch512，FLASHINFER，FULL_DECODE_ONLY CUDA Graph，前缀缓存开启，MTP关闭，纯文本模式。
- 总NVML显存监督预算40GiB，每100ms采样；监督器异常或超预算时停止自己的服务。

配置：[gpu3-abliterated-vocab4-40g-c2-256k.json](../deploy/nvidia/gpu3-abliterated-vocab4-40g-c2-256k.json)。运行证据：`outputs/services/gpu3-abliterated-vocab4-256k-20261107/`，包含进程、实际就绪配置、显存统计及通过服务器IP发起的两路功能检查。

实际分配KV池在256K下的驻留容量满足两路，并完成两路短请求检查。本次没有重复两路224K输入加完整32K输出的容量压力测试，不将缓存池计算视为该压力测试已通过。

重启先停止当前监督器、确认GPU释放，再把配置run_dir改为新目录并执行：

```bash
export QWEN_NVIDIA_ENV=runtime QWEN_INSTANCE=abliterated-vocab4-256k-c2-20261107
source scripts/env_nvidia.sh
poetry run python -m scripts.serve_nvidia_budget deploy/nvidia/gpu3-abliterated-vocab4-40g-c2-256k.json
```

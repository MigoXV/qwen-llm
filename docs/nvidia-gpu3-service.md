# GPU 3 原版 VOCAB4 量化服务

2026-11-03 部署，使用原版 Qwen3.8-27B 的 AWQ/GPTQ + VOCAB4 量化产物。

- 地址：`http://192.168.0.223:8132/v1`；监听 `0.0.0.0:8132`。
- 模型名：`qwen38-24g`；物理 GPU 3，TP1。
- 权重：`/workspace/opusi/Qwen/Qwen3.8-27B-W4A16-AWQ-GPTQ-VOCAB4-NVIDIA`。
- 沿用旧 GPU2 参数：总上下文 131072，最大输出 32768，最大输入 98304，并发 2，batch 512，固定 KV 8.75 GiB、FP8 E4M3，FLASHINFER，CUDA Graph，前缀缓存开启，MTP 关闭，GPU utilization 0.6。
- 沿用 24 GiB 总 NVML 显存监督器，每 100ms 采样，超额或监控异常时关闭自己的服务。采样监控不是硬件分区。
- 当前启动及双请求功能检查显存峰值约 23.64 GiB；通过服务器 IP 同时完成非思考、low 思考请求，服务归属确认为 GPU3。没有重新执行完整 128K 容量压力测试。

配置：[gpu3-original-vocab4-24g.json](../deploy/nvidia/gpu3-original-vocab4-24g.json)。监督器新增可选 `port` 配置，未指定时保持原有 `8130 + GPU` 规则。

进程、就绪信息、显存采样与请求原始结果：`outputs/services/gpu3-original-vocab4-20261103/`。

模型的完整非思考质量评估仍有一道固定题退化，单题思考模式复测通过；详见 `outputs/original-20261103-gpu2/README.md`。本次服务参数包含 FP8 KV，和此前使用 BF16 KV 的完整质量评估配置不同。

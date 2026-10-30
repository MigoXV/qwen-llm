# 单卡 24 GiB：词表量化、容量与质量实测

2026-10-29，分支 `feature/nvidia-quantization`。原始明细、配置与文件 SHA256 见 [机器可读结果](nvidia-24g-hotspots-results.json)。此前完整质量与长上下文矩阵仍以 [原报告](nvidia-w4a16.md) 为准，本轮采用用户要求的简短测试。

本轮有效改进是继续压缩词嵌入和输出层：模型加载占用从 **16.74 GiB 降到 13.23 GiB**。新版本单 A100、FP8 KV、256K 配置实测显存峰值 **23.419 GiB**；1–4 路短请求总吞吐 **67.33 / 129.91 / 190.17 / 249.54 tokens/s**。248K 输入检索通过。两种手写 Triton 实现都完成了数值与性能对照，但慢于原生 Marlin，未接入服务。

## 测量口径

- 硬件是 A100 PCIe **40 GB**，在总 NVML 显存超过 **24 GiB** 时停止本实验实例，每 100 ms 采样。不是实体 24 GB 显卡测试，也不是 CUDA 硬隔离，不能排除采样间的更短尖峰。
- 原词表 BF16 版本使用 GPU 3，新词表 INT4 版本使用 GPU 2；手写内核微基准在 GPU 3 服务退出后进行。没有把同一完整测试在两张卡上各跑一次。
- vLLM 0.32.0、Torch 2.13.0、compressed-tensors 0.17.0；TP1、BF16 激活、FP8 E4M3 KV、FLASHINFER、FULL_DECODE_ONLY CUDA Graph、batch tokens 512、并发上限 4、前缀缓存关闭、MTP 关闭。
- FP8 KV 仍是实验设置，沿用未单独校准的 1.0 缩放系数。GDN 状态继续使用原 BF16/FP32 精度。
- TPS 复用项目原 HTTP/SSE 计时器：同一成语提示，greedy，每路实际输出 512 token，每个并发预热后测 3 轮。总 TPS 包含首字等待。Profiler 请求独立执行，不进入 TPS 结果。
- 两套服务的 KV 预算和最大上下文也发生变化，且物理卡不同。约 7%–8% 的 TPS 差异是本轮配置观察值，不能当作严格控制变量的量化因果估计；算子微基准则是在同一卡上直接比较。

## 显存压缩与产物

新产物：

`/workspace/model-bin/windowsxp811203/Qwen3.8-27B-Abliterated-W4A16-AWQ-GPTQ-VOCAB4-NVIDIA`

来源为原 `W4A16-AWQ-GPTQ-NVIDIA`。主干已有 AWQ/GPTQ packed 权重保持不变；只对以下原 BF16 张量进行 group128 对称 INT4 RTN：

| 张量 | 形状 | 原字节数 | 量化后字节数（含 scale/shape） |
|---|---|---:|---:|
| `lm_head.weight` | 248320 × 5120 | 2,542,796,800 | 655,564,816 |
| `model.language_model.embed_tokens.weight` | 248320 × 5120 | 2,542,796,800 | 655,564,816 |

合计减少 **3.515 GiB**。导出对主分片其余 **1,982 个张量逐值核对**，另保留 MTP 分片。MTP 未启用，因此没有将其文件大小计为本次运行显存收益。

词表 RTN **没有使用校准数据**，不能称为 AWQ/GPTQ 校准后的词表。输出层、词嵌入的权重相对 L2 误差分别约 0.1249、0.1180；这是权重误差，不是能力下降百分比。

vLLM 已有 compressed-tensors 词嵌入查表反量化内核，但此版本的 `Qwen3_5Model` 没有向词嵌入传递量化配置与前缀。新增 [两行接线补丁](../deploy/nvidia/patches/qwen35-embedding-quant.patch)，由 [版本及 SHA256 清单](../deploy/nvidia/patches/qwen35-embedding-quant.json) 保护。没有量化词嵌入的模型仍走原浮点路径。

## 单卡 TPS

| 并发 | 原词表 BF16：总 TPS | 新词表 INT4：总 TPS | 新版平均首字延迟 | 新版平均单请求 decode TPS |
|---:|---:|---:|---:|---:|
| 1 | 63.06 | **67.33** | 90.97 ms | 68.01 |
| 2 | 120.85 | **129.91** | 134.73 ms | 66.06 |
| 3 | 176.83 | **190.17** | 162.12 ms | 64.66 |
| 4 | 231.75 | **249.54** | 169.33 ms | 63.66 |

总 TPS 是各请求实际输出 token 之和除以该轮墙钟时间；四路 249.54 不代表每路都是这个速度。三轮原始值、响应、输入输出长度、进程归属保存在 `outputs/nvidia-24g-20261029/{final55,vocab4-final}/short-tps.json`。

## 上下文与 24 GiB 预算

| 项目 | 原词表 BF16 | 新词表 INT4 |
|---|---:|---:|
| 配置的 KV 预算 | 5.5 GiB | 8.5 GiB |
| 实际 KV 池字节数 | 5,857,345,536 | 9,094,299,648 |
| 配置总上下文 | 168K | **256K** |
| 启动及测试总显存峰值 | 23.927 GiB | **23.419 GiB** |
| 实测精确输入长度 | 163840（160K） | **253952（248K）** |
| 中间校验码检索 | 通过，输出 12 token | 通过，输出 11 token |
| 首字延迟 | 111.21 s | **224.39 s** |
| 缓存抢占增量 | 0 | 0 |

K 按 1024 token 计，总上下文包含输入与输出。这里证明了启动、实际长输入处理和一次检索，**没有跑满剩余 8192 输出 token，也没有重新做所有并发的完整容量边界测试**。短检索输出的流式包会影响 decode 计时，不用它报告长上下文持续生成 TPS。

下表由实际分配池和 vLLM 自身的混合 Attention/GDN 页算法计算，精度为 1K；只是同时驻留的**容量估算**：

| 同时请求数 | 原版每路总上下文估算 | 新版每路总上下文估算 | 新版预留 8K 输出后的每路输入 |
|---:|---:|---:|---:|
| 1 | 168K | 256K | 248K |
| 2 | 82K | 130K | 122K |
| 3 | 53K | 85K | 77K |
| 4 | 38K | 62K | 54K |

因此，24 GiB 预算下的新版本具备单路 256K 配置和长输入实测证据；两路各 256K 的池预算不够。两路约 128K 的估算尚需完整生成验证，不能直接写成已认证容量。

### 冷启动限制

本轮采用两阶段流程，**两个阶段都没有超出 24 GiB 采样预算**：先用自动 KV 估算生成编译缓存，再命中相同缓存、按固定 KV 预算启动。

- 原版本直接冷启动固定 4.25 GiB/4.75 GiB KV 的试验峰值超过 25 GiB，被监测器终止。减少 batch tokens 或在 profiling 后调用 `gc.collect/empty_cache` 没有解决，清理代码已撤销。
- 原版本自动预算预编译阶段峰值约 20.69 GiB，因 KV 不足启动失败；随后固定 5.5 GiB、168K 启动成功。
- 新版自动预算预编译峰值约 17.18 GiB，报告可用 KV 7.41 GiB，小于 256K 所需约 8.18 GiB，按容量检查退出。随后固定 8.5 GiB、命中缓存启动成功。
- **没有证明新版在全空编译缓存下直接固定 8.5 GiB 也能启动。** 不要把 warm-cache 成功改写为一次冷启动成功。

实际命令、缓存目录、启动器源码哈希和峰值均在对应 `process.json`、`complete.json` 中。新版启动使用：

```bash
export QWEN_NVIDIA_ENV=runtime QWEN_INSTANCE=vocab4-limit24-20261029
source scripts/env_nvidia.sh
poetry run python scripts/patch_nvidia.py
# 预编译阶段：保留该 cache，预计因自动 KV 容量不足而退出。
CUDA_VISIBLE_DEVICES=2 poetry run python -m tests.performance.nvidia.cli serve \
  --model-path /workspace/model-bin/windowsxp811203/Qwen3.8-27B-Abliterated-W4A16-AWQ-GPTQ-VOCAB4-NVIDIA \
  --model-name qwen38-24g --port 8132 --tp 1 --context 262144 \
  --batch-tokens 512 --gpu-memory-utilization 0.6 \
  --kv-dtype fp8_e4m3 --attention-backend FLASHINFER --cuda-graph
# 进程释放后，重复同一命令并增加 --kv-cache-gib 8.5。
```

这些是服务参数，`gpu-memory-utilization` 在显式 KV 预算下不等于整卡硬限制。本轮还由 `outputs/nvidia-24g-20261029/serve_budget.py` 独立采样、监督总显存并在超限时退出，复现实验需要保留这个监控条件。

## 质量与采用范围

| 检查 | 新词表 INT4 | 原 FP8 KV 版本相同题目 |
|---|---:|---:|
| 固定题 | 40/40 | 40/40 |
| 项目代码题 | 64/64 | 64/64 |
| GSM8K 前 32 题 | 31/32 | 31/32 |

136 题逐题通过状态无变化，题目、数据集和代码判题器一致。对照来自历史完整评估，**没有重跑 1319 题完整质量门槛，也没有证明能力下降小于 1 个百分点**。词表版本暂作为显存优先的实验产物。

20 道正常敏感提示的拒答启发式标记为 0。人工检查仍看到事实错误与论点混淆，历史原始/量化版本也有同类问题；该检查不是事实准确率，不能把“不拒答”解释成“无质量损失”，也不足以把错误归因于本次量化。

## 架构与交付边界

延续 [平台架构审查](nvidia-architecture.md)：HTTP/SSE 计时与判题器共享；词表导出配方和验证工具归 NVIDIA 实验模块；产物 `config.json` 是精度方案的权威来源，manifest 保存来源和验证范围；NVIDIA runtime 补丁只负责已有量化能力的模型接线。没有新增通用算子注册框架或修改 Ascend 算法。

配置、容量与判题器相关检查 14 项通过。原 Ascend 环境和原模型目录没有改写。本报告对应的测试进程已退出；后续启动的 [GPU 2 常驻服务](nvidia-gpu2-service.md)继续运行。

服务采用 vLLM 原生 Marlin。未采用的手写算子、专用 profiling 接口和中间数据已按用户要求删除；本报告仅保留实际采用的量化方案与验证结果。历史算子实验如需追溯，可查 Git 提交 `ba88ee0`，无需在工作区维护。

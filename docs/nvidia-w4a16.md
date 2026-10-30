# NVIDIA Qwen3.8-27B 量化与 vLLM 实测

状态：2026-10-29 完成容量、短请求 TPS、全部短题质量、长文检索、8K 冷热缓存和 3632 秒持续负载验证。双卡 BF16 KV 为主方案；单卡 FP8 KV 保持实验状态。持续负载中观察到的主机 RSS 增长，后续已定位至 vLLM 通知队列并提供 NVIDIA 环境补丁，见 [RSS 定位报告](nvidia-rss.md)。本页性能及质量数值保留修复前的原始结果。

原模型和 Ascend 的 ModelSlim 量化、专项 Worker、启动算法保持原样。NVIDIA 新增隔离环境和同级 Engine，主产物采用 W4A16 AWQ+GPTQ，GPTQ-only 作为独立对照。两份权重文件各合计 **18.12 GiB**，没有裁剪语言骨干。

硬件为 A100 PCIe 40GB：物理 GPU 2、3 组成双卡 TP2 主实例，物理 GPU 1 同时运行单卡实验。FP8 只用于实验 KV 存储，常规入口保持 BF16 KV；GPU 0 未使用。

| 并发请求数 | 双卡 AWQ＋BF16 KV | 单卡 AWQ＋FP8 KV 实验 |
| --- | ---: | ---: |
| 1 | 256K | 256K |
| 2 | 256K | 256K |
| 3 | 256K | 184K |
| 4 | 208K | 136K |

表内为每请求最大完整通过总上下文，**包含 8K 输出**，K = 1024；双卡每卡固定 26.5 GiB KV，单卡固定 18 GiB KV。每点均实际输出满 8192 token、无抢占，未达到原生上限的边界按 8K 步长确认，下一档实测发生抢占。原生上限仍是 262144，没有修改 RoPE 或外推上下文。

同 32K 配置、固定对端负载的 512-token 输出对照中，双卡量化吞吐为原 BF16 的 **1.88–1.97 倍**。GSM8K 由 96.59% 变为 96.13%，下降 0.455 个百分点，固定题和代码题均保持全过。单卡同后端短请求对照没有看到 FP8 KV 的速度提升；它在本次测试中的主要收益是容量。

全部原始证据位于 `outputs/nvidia-20261028/`，随代码保存的数字摘要见 [实测结果 JSON](nvidia-w4a16-results.json)。表格按各实例自身的实际输出 token 和请求耗时计算，未使用 SSE 包数替代 token 数。长文检索各 45/45、8K 热缓存检索各 90/90；最终逐项证据核对为 `final-acceptance.json`，200 项检查通过。

## 已确认的容量边界：双卡

每卡显式 KV 预算 26.5 GiB，来自全新编译缓存；BF16 KV、batch 2048、无前缀缓存、无 MTP。下表每个通过点均实际完成每请求 8192 输出 token，精确输入长度、实例身份、设备归属核验通过，抢占增量为 0。每点一次完整压力测试，K = 1024；总上下文包含输入和输出。

| 并发 | 最大通过总上下文 | 每路输入 | 最小失败总上下文 | 总输出 TPS | TTFT 中位数秒 | 每请求 decode TPS 中位数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 256K | 248K | 达到模型原生上限 | 27.836 | 137.01 | 52.08 |
| 2 | 256K | 248K | 达到模型原生上限 | 33.162 | 211.14 | 31.19 |
| 3 | 256K | 248K | 达到模型原生上限 | 33.733 | 285.38 | 18.58 |
| 4 | 208K | 200K | 216K：真实抢占 | 46.072 | 267.23 | 18.85 |

四路 216K 发生一次抢占后立即取消并排空，没有将重算或排队完成当作显存同时容纳。边界分辨率为 8K，原始通过点和失败点均在 `fixed-main-capacity/`；派生统计为 `fixed-evidence-summary.json`。长请求 TPS 包含预填充；首增量到结束的 decode 统计也包含调度停顿，不能当作纯算子速度。

因此，在这组固定预算下，两张卡的容量可支持 1–3 路各 256K，四路各 128K 也低于已完整验证的 208K 边界。四路各 256K 不在该容量内。长文检索另行完成，各档结果见下文。

## 已确认的容量边界：单卡 FP8

显式 KV 预算 18 GiB，FLASHINFER、FP8 E4M3，其余容量判据与双卡一致。完整矩阵已完成；20 GiB 冷启动失败记录保留，因此此表描述可从全新缓存启动的 18 GiB 配置，不称作任何预算下的显卡物理极限。

| 并发 | 最大通过总上下文 | 每路输入 | 最小失败总上下文 | 总输出 TPS | TTFT 中位数秒 | 每请求 decode TPS 中位数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 256K | 248K | 达到模型原生上限 | 19.994 | 222.33 | 43.72 |
| 2 | 256K | 248K | 达到模型原生上限 | 23.700 | 335.80 | 25.61 |
| 3 | 184K | 176K | 192K：真实抢占 | 38.116 | 262.68 | 21.54 |
| 4 | 136K | 128K | 144K：真实抢占 | 56.249 | 207.00 | 22.19 |

三路 192K 和四路 144K 均观察到抢占，取消后排空，再继续测试；表中通过点的抢占增量均为 0。原始记录在 `fixed-single-capacity/`，两实例合并表为 `fixed-capacity.csv`。各行使用标注的上下文长度，不能忽略长度差异直接比较 TPS。

两实例随后以相同预算重启为自然停止模式，命中 AOT 缓存；实际 KV storage 字节数与所有长度的池容量均与冷编译启动完全一致，记录为 `fixed-restart-reproducibility.json`。这验证了固定预算的容量可复现性；后续检索和持续负载均使用该自然停止配置。

## 固定 256K 配置的短请求 TPS

两实例均配置 262144 上限，双卡每卡 26.5 GiB KV、单卡 18 GiB KV；每请求实际输出 512 token、各并发 5 轮、每轮同步开始。以下为中位数，均无错误或抢占。量化相对原权重的增益仍使用前面的同 32K 配置对照。

| 并发 | 双卡 AWQ 总 TPS | 单卡 AWQ FP8 KV 总 TPS | 双卡 TTFT ms | 单卡 TTFT ms | 双卡每请求 decode TPS | 单卡每请求 decode TPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 76.989 | 62.335 | 110.58 | 97.63 | 78.30 | 62.94 |
| 2 | 146.405 | 119.213 | 160.43 | 145.52 | 75.01 | 60.63 |
| 3 | 204.369 | 174.913 | 207.55 | 194.49 | 69.97 | 59.52 |
| 4 | 269.070 | 229.514 | 211.77 | 197.30 | 69.10 | 58.57 |

记录为 `fixed-paired-generation/` 与对应的 `fixed-paired-generation-instance-*-summary.json`，前后 Prometheus 抢占计数增量为 0。

## 原 BF16 与量化的同配置 TPS 对照

两次测试均使用物理 GPU 2、3、TP2、BF16 KV、32K 服务上限、batch 2048、显存预算 0.9、FULL_DECODE_ONLY、关闭前缀缓存和 MTP。GPU 1 上保持同一个 AWQ FP8 KV 实例，每轮两边同步发起相同并发和输出长度。表内为每请求实际输出 512 token、每档 5 轮的单实例总 TPS 中位数；全部请求完整、无失败，模型和实例身份核验通过。

| 并发 | 原 BF16 总 TPS | AWQ 总 TPS | 吞吐比 | 原 BF16 TTFT ms | AWQ TTFT ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 38.871 | 76.473 | 1.97× | 95.55 | 109.04 |
| 2 | 74.986 | 146.351 | 1.95× | 131.50 | 117.74 |
| 3 | 107.612 | 203.835 | 1.89× | 174.31 | 128.68 |
| 4 | 142.366 | 268.058 | 1.88× | 180.83 | 219.70 |

量化提高了这组持续输出吞吐，短提示的首 token 延迟并非每档都降低。TTFT 为 HTTP 发起到首个非空增量的时间中位数；总 TPS 为本轮该实例的实际输出 token 总数除以自身请求完成时间。原始记录为 `paired-awq-fp8-generation/` 和 `paired-original-fp8-generation/`，配置及对端实例一致性核验与计算结果为 `paired-main-comparison.json`。

## 128-token 口径与单卡 KV 精度速度对照

另保留与 Ascend 旧流程一致的短输出口径：相同短提示、128 输出 token、每档 3 轮中位数，服务上限同为 32K，双实例每轮同步。所有请求实际输出 128 token、无失败。

| 并发 | 双卡原 BF16 总 TPS | 双卡 AWQ 总 TPS | 同期单卡 AWQ FP8 KV 总 TPS |
| --- | ---: | ---: | ---: |
| 1 | 38.378 | 73.978 | 60.998 |
| 2 | 72.505 | 135.653 | 113.537 |
| 3 | 104.074 | 188.693 | 165.461 |
| 4 | 137.565 | 248.749 | 215.633 |

原始记录为 `paired-original-fp8-short/`、`paired-awq-fp8-short/`。下表的 KV 精度对照则使用 512-token 口径。

单卡 AWQ、FLASHINFER、32K 上限和其余参数相同，每请求实际输出 512 token、每档 5 轮；对端均为相同配置的双卡 AWQ 实例，轮次同步开始。切换 KV 配置时对端重启，不能称为同一 PID。全部请求完整、无失败。

| 并发 | BF16 KV 总 TPS | FP8 KV 总 TPS | BF16 KV TTFT ms | FP8 KV TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 62.422 | 61.857 | 99.20 | 99.84 |
| 2 | 120.200 | 119.333 | 144.82 | 148.92 |
| 3 | 176.764 | 175.081 | 187.73 | 193.48 |
| 4 | 231.452 | 229.379 | 201.56 | 199.46 |

这组短提示下 FP8 KV 吞吐约低 0.7%–1.0%，没有实测速度收益；容量由前面的独立长请求测试验证。匹配条件与原始计算为 `paired-single-kv-comparison.json`，原始轮次为 `paired-awq-fp8-generation/` 和 `paired-awq-bf16kv-generation/`。

## 完整短题质量与行为对照

每种配置均完成相同的 1423 个请求：40 道固定题、64 道项目代码题和 GSM8K test 全集 1319 题。greedy、非思考；数学输出上限 2048，固定题 256，代码题 512。数据集、提示政策和代码判题器哈希一致，所有配置执行错误为 0，实例身份未变化。

| 配置 | 固定题 | 项目代码题 | GSM8K test | 相对原 BF16 |
| --- | ---: | ---: | ---: | ---: |
| 原 BF16，TP2，BF16 KV，FLASH_ATTN | 40/40 | 64/64 | 1274/1319（96.59%） | 基线 |
| AWQ+GPTQ，TP2，BF16 KV，FLASH_ATTN | 40/40 | 64/64 | 1268/1319（96.13%） | −0.455 个百分点 |
| GPTQ-only，TP2，BF16 KV，FLASH_ATTN | 40/40 | 64/64 | 1267/1319（96.06%） | −0.531 个百分点 |
| AWQ，TP1，BF16 KV，FLASHINFER | 40/40 | 64/64 | 1272/1319（96.44%） | −0.152 个百分点 |
| AWQ，TP1，FP8 KV，FLASHINFER | 40/40 | 64/64 | 1268/1319（96.13%） | −0.455 个百分点 |

全部通过原项目门槛：原来通过的固定题不新增失败，代码通过数不减少，完整 GSM8K 下降不超过 1 个百分点。代码题不是 HumanEval。AWQ 双卡数学题有 13 道原通过题变为失败，7 道原失败题变为通过，净少 6 道；量化并非无损。

单卡同 AWQ 权重、同 FLASHINFER 对照中，FP8 KV 相对 BF16 KV 净少 4 道数学题，下降 0.303 个百分点，也通过门槛。该 BF16 KV 服务上限为 128K、FP8 为 256K，实际质量题均低于两者；不能用答题总耗时比较 TPS。AWQ 双卡和单卡 FP8 总分相同，但有 14 道数学题的通过/失败结果不同，不能据此称 FP8 无损。

AWQ 是既定主候选；GPTQ 对照只比 AWQ 少答对 1 道数学题，不能据此宣称普遍显著优劣。完整 TPS、容量、检索和稳定性均使用 AWQ，GPTQ 不套用这些长上下文结论。

BF16、AWQ、GPTQ 的 20 道正常敏感提示均未出现启发式拒答标记。人工抽查发现个别论述存在观点对应或事实问题；这组只比较拒答变化，不作为事实正确率或普遍无拒答结论。文本、JSON 约束、工具调用和流式断连后的并发槽回收探针均已通过。

原始回复和逐题判定：`bf16-quality.json`、`awq-quality.json`、`gptq-quality.json`、`awq-single-bf16-quality.json`、`awq-single-fp8-quality.json`；门槛为各 `*-quality-gate.json` 及 `awq-single-fp8-vs-bf16-quality.json`。汇总 CSV 为 `quality-comparison.csv`，行为文本为 `*-behavior.json`。

## 长文检索

两实例各 **45/45**，精确输入长度、实例身份和设备归属检查通过，无请求错误或抢占。自然停止服务与容量服务分开启动，检索不使用强制长度或前缀缓存。单路每档输入覆盖开头／中部／末尾各 3 份不同校验码材料；2–4 路在已通过的最大上下文输入长度下，各位置测一轮，每个请求使用不同校验码。

| 并发 | 双卡精确输入长度 | 双卡命中 | 单卡 FP8 精确输入长度 | 单卡命中 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 120K、248K | 18/18 | 120K、248K | 18/18 |
| 2 | 248K | 6/6 | 248K | 6/6 |
| 3 | 248K | 9/9 | 176K | 9/9 |
| 4 | 200K | 12/12 | 128K | 12/12 |

这里的长度仅指输入；每请求最多输出 32 token，按实际自然停止答案逐字核对校验码。3 份材料使用不同校验码，解码 seed 固定为 42。原始记录逐轮保存在 `fixed-{main,single}-retrieval-c*/`。该测试是单个校验码检索，不代表所有长文任务。

## 8K 冷热缓存

两实例统一为 32K 服务上限，仍采用上述固定 KV 预算。精确 8192 输入＋实际 128 输出，1–4 路全部完成。冷请求关闭前缀缓存，每档 3 轮；热请求另启用前缀缓存，每档 7 轮，以下汇总最后 3 轮中位数。两实例同时执行各自矩阵，本项不使用逐轮同步屏障。

| 并发 | 双卡冷请求总 TPS | 双卡热请求总 TPS | 单卡 FP8 冷请求总 TPS | 单卡 FP8 热请求总 TPS |
| --- | ---: | ---: | ---: | ---: |
| 1 | 27.001 | 66.669 | 25.278 | 52.945 |
| 2 | 33.461 | 117.438 | 31.061 | 99.187 |
| 3 | 35.809 | 155.773 | 34.129 | 136.792 |
| 4 | 37.587 | 193.734 | 35.456 | 171.791 |

| 并发 | 双卡冷 TTFT ms | 双卡热 TTFT ms | 单卡冷 TTFT ms | 单卡热 TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2993.01 | 270.30 | 3017.51 | 299.44 |
| 2 | 4453.02 | 368.45 | 4520.94 | 371.17 |
| 3 | 6600.08 | 556.42 | 6740.41 | 600.31 |
| 4 | 8033.60 | 707.42 | 8476.82 | 722.54 |

冷请求实际 cached tokens 均为 0；热请求纳入统计的各请求均为 7840。所有请求输出满 128 token，无错误或抢占。热缓存检索另测每实例 30 份不同材料、每份重复 3 次，均为 **90/90**；复用轮次均观察到实际缓存命中，无请求间校验码串扰。

原始记录为 `fixed-{main,single}-8k-{cold,hot,hot-retrieval}/`。首轮和后续轮次均保留，热性能只取最后三轮；不将 90 次重复检索称为 90 份独立材料。

前缀缓存的 GDN 状态存储会改变容量，所以 32K 热缓存结果不套用到前面关闭前缀缓存的 256K 配置。

## 持续稳定性

两个固定预算的 256K 服务同步重复 1–4 路、512-token 输出矩阵 **3632.37 秒（60.54 分钟）**，共 83 批。每实例 830 个正式请求全部输出满 512 token，另外每批各一次预热不计入 TPS；所有批次有效，实例身份不变，错误和前后抢占增量均为 0。关闭前缀缓存，MTP 为 0。

| 并发 | 双卡总 TPS 中位数 | 单卡 FP8 总 TPS 中位数 | 双卡 TTFT ms | 单卡 TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 77.352 | 61.949 | 105.42 | 92.48 |
| 2 | 146.300 | 119.366 | 120.07 | 146.41 |
| 3 | 204.142 | 175.078 | 210.13 | 186.88 |
| 4 | 268.884 | 229.692 | 213.21 | 189.99 |

各档末 10 批相对首 10 批的 TPS 中位数变化均小于 0.13%。NVML 采样的双卡显存各固定为 36921 MiB，单卡固定为 37295 MiB；所有进程的末次 FD 数与起始相同，GPU 0 的观测占用始终为 0。双卡采样温度分别为 29–54°C、30–52°C，单卡为 56–68°C；时钟、功耗及逐进程数据均保留。

**修复前的主机内存增长记录**：从第 1 批结束到第 83 批结束，Worker RSS 增长 8.13 MiB，Engine Core 增长 8.15 MiB，最后 20 批各仍增长约 1.96 MiB；API RSS 已稳定。双卡各进程最后 20 批 RSS 不变。后续在物理 2、3 号卡定位到 vLLM 0.32.0 `SpinCondition` 忙循环期间 ZeroMQ 通知积压，BF16 KV 也可复现；已在独立 NVIDIA runtime 中增加定期 EVENTS 查询，CPU 因果对照与 GPU 验证见 [后续报告](nvidia-rss.md)。这次一小时结果仍属于修复前环境，不能当作修复后的长期无泄漏证明；FP8 KV 继续保留为实验。

这里是短请求持续负载，长上下文压力由前面的完整生成测试覆盖。原始记录为 `fixed-dual-stability/`，资源范围汇总为 `fixed-evidence-summary.json`，首尾趋势为 `stability-resource-trend.json`。测试完成后服务及所属 Worker 全部退出，8120、8121 端口释放，GPU 上无模型计算进程。

## 量化产物与完整性

主产物：`/workspace/model-bin/windowsxp811203/Qwen3.8-27B-Abliterated-W4A16-AWQ-GPTQ-NVIDIA`。对照产物：同目录下 `Qwen3.8-27B-Abliterated-W4A16-GPTQ-NVIDIA`。两者均附中文 README、配方、清单和许可证。

两种配方使用相同的 512 条校准材料，最大 4096 token，固定 open-perfectblend revision `af60f3c18201652a83a93f46fcfee1b646ba3df7`；实际输入 JSONL SHA256 为 `5ae6096f159293e7da3fed4ab34bd3107f15c202eeb18815c84bb0ef7a4630ca`。AWQ 激活缓存卸载到 CPU，随后 GPTQ；另一候选只执行 GPTQ。校准及导出分别约 102.1、28.4 分钟。

每个候选有 400 个 Linear 使用对称 INT4 group128、BF16 激活，compressed-tensors 格式，实际由 Marlin 执行。视觉模块、词嵌入、输出头、GDN 的 a/b 投影及 MTP 保留浮点。每个候选两个权重文件合计 19,452,784,112 字节（18.12 GiB），包含本次文本推理未使用的视觉和 MTP 权重。

原模型 1199 个张量均有对应项或量化替代，导出共 1999 个张量；没有遗漏原语言骨干。15 个 MTP 张量逐字节一致，分词器、聊天模板、生成配置、消融元数据和许可证哈希一致。模型配置仅新增 `quantization_config`。分片哈希与清单匹配，原模型的事前本地 SHA256 快照在量化后复核不变；这不等同于上游发布者签名验证。

记录为 `artifact-awq.json`、`artifact-gptq.json`、`source-manifest.json`。原 BF16 每卡加载约 25.15 GiB，AWQ 双卡每卡约 8.44 GiB，AWQ 单卡约 16.74 GiB。推理只加载完整语言骨干，未测试视觉能力或 MTP 推测解码，所有本次性能结果 MTP 为 0。

FP8 KV 实际后端为 FLASHINFER，所有 K/V scales 为默认 1.0，未做 KV 校准。只有 full-attention KV 使用 FP8，GDN 状态仍为 BF16／FP32；底层缓存 int8/uint8 tensor 是字节视图。[vLLM FP8 KV 文档](https://docs.vllm.ai/en/v0.32.0/features/quantization/quantized_kvcache/)

## 显式 KV 预算与冷启动复验

同为 0.9 显存预算、262144 上限、batch 2048、无前缀缓存和 MTP，首次编译与直接加载 AOT 缓存时的内存 profile 不同。实际双卡每卡 KV 池从 26,255,294,464 变为 28,695,855,104 字节，单卡 FP8 从 14,900,264,960 变为 19,524,485,120 字节。后者对应完整 256K 的池容量由 1.70 路变为 2.22 路，但容量计算本身不是完整生成通过证据。

日志证据为 `awq-single-service.log` 与 `capacity-single-fp8-service.log`：首次编译记录 consumed 19.11 GiB、peak activation 2.52 GiB；加载 AOT 记录 16.92 GiB、0.41 GiB。vLLM 自动预算会扣除这些 profile 数值，因此仅固定显存比例不能固定本次启动的实际容量。已改用显式预算并完成全新编译缓存的启动与完整生成复验，原缓存和原始记录保留。

自动 0.9 预算下，两实例的单路 128K、192K、256K 均完成全部 8192 输出 token，无抢占；双卡另完成双路 128K。发现启动差异后，剩余中间点扫描主动中止，转为显式预算验收全部 1–4 路。原始 `capacity-main/`、`capacity-single-fp8/` 保留，中断轮次不作为通过或容量失败证据，原因见 `automatic-capacity-superseded.json`。

全新编译缓存下，双卡每卡 26.5 GiB KV 成功启动，实际池每卡 28,438,953,984 字节；单卡 20 GiB KV 在分配时 OOM：请求 19.96 GiB、当时只有 19.58 GiB 空闲。日志中的 CUDA 0 是物理 GPU 1 在该进程内的编号。失败实例正常释放后，单卡 18 GiB 在另一个全新缓存下启动成功，实际池为 19,318,964,224 字节。这两个固定预算的完整生成测试均已完成；单卡 20 GiB 冷启动结果不能被缓存命中后可能成功的启动掩盖。

## 环境、架构与回归

- 原始模型：`/workspace/model-bin/windowsxp811203/Qwen3.8-27B-Abliterated`，18 个 safetensors 分片，保持只读；本地 SHA256 快照为 `outputs/nvidia-20261028/source-manifest.json`。
- 模型架构：Qwen3.5 dense，64 层中 16 层 full attention、48 层 Gated DeltaNet，原生配置最大上下文 262144。
- GPU：NVIDIA A100 PCIe 40GB，主实例为物理 2、3，单卡实验为物理 1，0 号卡未使用。GPU 2、3 无 NVLink。
- 驱动 595.71.05；CUDA 推理使用 Python 3.11、torch 2.13.0+cu130、vLLM 0.32.0、transformers 5.17.0、compressed-tensors 0.17.0。
- 量化使用独立环境：llmcompressor 0.14.0.1、compressed-tensors 0.19.0、FLA 0.6.0。两环境 `pip check` 通过。完整版本清单分别保存于 `/workspace/qwen-nvidia/{runtime,quant}/installed.txt`。

环境及实验命令见 [测试说明](../tests/performance/nvidia/README.md)，职责边界见 [架构审查](nvidia-architecture.md)。本机无法执行 NPU 硬件回归；Ascend 原启动方法相对 `3a03213` 的 AST 完全一致。18:47 UTC 前的完整 CPU/协议回归：原 Python 3.10 环境为 208 passed、7 skipped，CUDA Python 3.11 环境为 229 passed、4 skipped；测试时隐藏所有 GPU。记录为 `original-regression-final.log` 和 `runtime-regression-final.log`。

## 复现启动

先确保目标 GPU 和端口空闲。主实例：

```bash
cd /workspace/opusi/qwen-llm
export QWEN_NVIDIA_ENV=runtime QWEN_INSTANCE=main
source scripts/env_nvidia.sh
CUDA_VISIBLE_DEVICES=2,3 NCCL_P2P_DISABLE=1 poetry run qwen-llm serve-nvidia \
  --model-path /workspace/model-bin/windowsxp811203/Qwen3.8-27B-Abliterated-W4A16-AWQ-GPTQ-NVIDIA \
  --model-name qwen38-awq --port 8120 --max-model-len 262144 \
  --kv-cache-gib 26.5 --batch-tokens 2048 --cuda-graph
```

单卡 FP8 KV 实验在另一终端启动：

```bash
cd /workspace/opusi/qwen-llm
export QWEN_NVIDIA_ENV=runtime QWEN_INSTANCE=single-fp8
source scripts/env_nvidia.sh
CUDA_VISIBLE_DEVICES=1 poetry run python -m tests.performance.nvidia.cli serve \
  --model-path /workspace/model-bin/windowsxp811203/Qwen3.8-27B-Abliterated-W4A16-AWQ-GPTQ-NVIDIA \
  --model-name qwen38-single-fp8 --port 8121 --tp 1 --context 262144 \
  --kv-cache-gib 18 --kv-dtype fp8_e4m3 --attention-backend FLASHINFER --cuda-graph
```

完整环境安装、校准、质量、同步 TPS、容量、检索及冷热缓存命令见 [测试说明](../tests/performance/nvidia/README.md)。实验入口的 `--force-length` 仅用于容量，不能用于自然停止的检索或普通服务。固定 KV 预算优先于自动显存比例，需连同实际池字节数记录。

## 故障与处理

1. 默认 NCCL P2P 卡在 communicator 初始化的 synchronize，使用 `NCCL_P2P_DISABLE=1` 并禁用自定义 all-reduce 后启动正常。两卡对照统一使用该设置。原始栈保存在 `nccl-hang-stack.txt`。
2. transformers 5 的聊天模板可能返回 Mapping；共享生成层兼容旧 list 和新 Mapping。原模板参数、采样和 Ascend 启动算法保持一致。
3. AWQ 校准的激活缓存默认放在 GPU，第 8 层发生 OOM。改用官方 AWQ `offload_device=cpu` 后通过该层，保留相同 512 条校准数据、4096 最大长度和配方。
4. GDN 校准起初使用 PyTorch 参考实现，安装 FLA 后使用上游优化内核。128/512/4096-token 随机张量对照的相对 L2 误差约 0.44%，最大绝对误差 0.000488；这属于 BF16 内核数值差异，不是模型质量验收结果。
5. 初次 streaming 数据集结束时 PyArrow 清理异常，已显式关闭迭代器。原 512 条 JSONL 的数量、哈希验证通过；修复后的 8 条复核任务正常退出。

失败或主动停止的量化目录保留 `.quantization.incomplete`，CUDA 入口拒绝加载；不能把这些目录当作可用量化模型。

## 早期基线记录

早期 BF16 Eager/Graph、8K 冷请求与 24K 检索记录保留。24K 检索为 90/90；Graph 512-token 总 TPS 为 38.626/74.717/106.845/140.892（1–4 路），8K 冷请求为 20.91/28.04/31.87/33.78。当时对端为校准任务，因此量化增益使用正文后补的固定对端矩阵，不使用这些早期数值归因。

原始文件为 `bf16-eager-short.json`、`bf16-graph-short.json`、`bf16-graph-generation.json`、`bf16-8k-cold/`、`bf16-retrieval-24k/`、`bf16-protocol.json`、`bf16-lifecycle.json`。所有失败或主动中止的量化目录保留 `.quantization.incomplete`，CUDA 入口拒绝加载。

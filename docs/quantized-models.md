# 四款 Qwen 量化模型适配与性能报告

2026-10-15，Ascend910B4-1，物理4/逻辑0，64 GiB。使用 CANN 9.1.1、torch/torch-npu 2.9.1、源码 vLLM 0.18.0 与 vLLM-Ascend 0.18.0，本地依赖和源码补丁见 [环境升级报告](cann911-torch291.md)。所有测试都在本机真实 NPU 上执行。

模型、CANN、新 Poetry 环境、构建与运行缓存均放在 `/home` 数据盘。模型位于 `/home/qwen-models/weights/`，项目 `model-bin/` 仅用符号链接；六份下载均核对固定文件 revision、大小和 SHA256，逐文件清单保留在模型目录，摘要见 [model-downloads.json](model-downloads.json)。所有原权重与失败输出均保留。

## ModelScope来源与存储

| 目录名 | ModelScope来源 | 磁盘/GiB |
| --- | --- | ---: |
| Qwen3-30B-A3B-Instruct-2507-W4A8 | [vllm-ascend/Qwen3-30B-A3B-Instruct-2507-quantized.w4a8](https://modelscope.cn/models/vllm-ascend/Qwen3-30B-A3B-Instruct-2507-quantized.w4a8) | 29.472 |
| Qwen3-30B-A3B-Instruct-2507-W8A8 | [vllm-ascend/Qwen3-30B-A3B-Instruct-2507-quantized.w8a8](https://modelscope.cn/models/vllm-ascend/Qwen3-30B-A3B-Instruct-2507-quantized.w8a8) | 29.091 |
| Qwen3.5-35B-A3B-W8A8 | [Eco-Tech/Qwen3.5-35B-A3B-w8a8-mtp](https://modelscope.cn/models/Eco-Tech/Qwen3.5-35B-A3B-w8a8-mtp) | 37.074 |
| Qwen3.5-9B-W8A8 | [sogagaga/Qwen3.5-9B-W8A8](https://modelscope.cn/models/sogagaga/Qwen3.5-9B-W8A8) | 13.068 |
| Qwen3.5-9B-W8A8-RedHatAI | [RedHatAI/Qwen3.5-9B-quantized.w8a8](https://modelscope.cn/models/RedHatAI/Qwen3.5-9B-quantized.w8a8) | 13.064 |
| Qwen3.6-35B-A3B-W8A8 | [Eco-Tech/Qwen3.6-35B-A3B-w8a8](https://modelscope.cn/models/Eco-Tech/Qwen3.6-35B-A3B-w8a8) | 37.077 |

## 默认配置与吞吐

TPS 为实际成功输出 token 数除以整个并发批次 makespan。短输入使用同一成语提示、128 输出 token，1/2/4 路各三轮取中位数；4路列为合计吞吐。8路验证四路容量的429拒绝。温度0、关闭思考。HBM是整卡轮次采样最大值，包含驱动基础占用，非连续峰值。

长输入每路精确8192输入+128输出。首轮冷缓存只测一次，最后三轮重复请求取中位数；各请求实际缓存命中数单独记录，不能把未命中缓存的重复请求称为热缓存。混合模型保留额外三轮过渡。

| 模型 | 1路TPS | 2路TPS | 4路TPS | 整卡HBM/GiB | 8K首轮TPS | 8K重复TPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3.6-35b-a3b | 63.13 | 101.14 | 188.25 | 43.31 | 68.24 | 138.83 |
| qwen3.5-35b-a3b | 62.62 | 100.66 | 184.73 | 42.18 | 63.75 | 137.30 |
| qwen3-30b-a3b-instruct-2507 | 67.24 | 137.90 | 270.00 | 39.81 | 106.82 | 192.87 |
| qwen3.5-9b | 75.50 | 130.01 | 243.38 | 19.38 | 93.04 | 167.98 |

## 能力与8K检索

固定40道基础题加GSM8K test前32题，严格检查指定答案和完成状态，保留每题原文。GSM8K来自OpenAI官方仓库固定提交 `3101c7d5072418e28b9008a6636bde82a006892c`，数据在 `/home/qwen-models/evaluation/`。72题是能力冒烟，不代表完整基准；没有运行完整BF16模型作参照，不能据此计算量化精度损失。

8K检索四路使用不同校验码，分别覆盖重复data填充与混合中文记录。

| 配置 | 基础题 | GSM8K | 总分 | 重复填充8K | 混合中文8K |
| --- | ---: | ---: | ---: | ---: | ---: |
| qwen3.6-35b-a3b | 40/40 | 31/32 | 71/72 | 4/4 | 4/4 |
| qwen3.5-35b-a3b | 40/40 | 32/32 | 72/72 | 4/4 | 4/4 |
| qwen3-30b-a3b-instruct-2507 | 40/40 | 32/32 | 72/72 | 0/4 | 4/4 |
| qwen3.5-9b | 40/40 | 31/32 | 71/72 | 4/4 | 4/4 |
| qwen3-30b-a3b-instruct-2507-w4a8 | 40/40 | 31/32 | 71/72 | 0/4 | 4/4 |
| qwen3.6-35b-a3b-mtp2 | 40/40 | 31/32 | 71/72 | 4/4 | 4/4 |
| qwen3.5-35b-a3b-mtp2 | 40/40 | 31/32 | 71/72 | 4/4 | 4/4 |

30B W8A8/W4A8都未通过重复data压力材料；混合中文材料通过。W8A8在eager模式下也未通过重复材料，因此不能只归因为图执行；未运行原始浮点权重，原因没有归因为量化本身。两份权重与失败输出均保留。

W4A8的唯一能力题失分为正确数值使用了加粗的FINAL行，违反指定格式；仍按既定规则计为未通过。9B与部分35B配置的GSM8K未通过项涉及盈亏平衡年份的解释或在2048输出token预算内未完成。

9B两步MTP额外将29×31算成897（正确899），总分70/72，虽然更快也不设为默认。3.5-35B两步MTP得分71/72，基线72/72，保留为可选配置。3.6-35B两步MTP本套判定与基线一致，但长输入缓存/显存取舍仍需考虑。

## 可选配置的性能取舍

| 配置 | 1路TPS | 4路TPS | HBM/GiB | 8K重复TPS |
| --- | ---: | ---: | ---: | ---: |
| qwen3-30b-a3b-instruct-2507-w4a8 | 75.36 | 248.75 | 28.27 | 179.36 |
| qwen3.6-35b-a3b-mtp2 | 71.83 | 209.78 | 46.22 | 104.19 |
| qwen3.5-35b-a3b-mtp2 | 74.68 | 213.15 | 46.22 | 103.81 |

这些配置提供不同速度、显存与能力结果，不替换默认配置。9B两步MTP因新增基础算术错误只保留原始试验记录。

## 参数与可选加速

| 配置名 | 精度 | KV/GiB | batch tokens | 图模式 | MTP步数 | 共享专家重叠 | 原始报告 |
| --- | --- | ---: | ---: | --- | ---: | --- | --- |
| qwen3.6-35b-a3b | W8A8 | 4 | 4096 | FULL_DECODE_ONLY | 0 | 开启 | [w8a8-overlap-b4096-kv4](/home/qwen-models/reports/qwen3.6-35b-a3b/w8a8-overlap-b4096-kv4) |
| qwen3.5-35b-a3b | W8A8 | 3 | 4096 | FULL_DECODE_ONLY | 0 | 关闭 | [w8a8-base-b4096-kv3](/home/qwen-models/reports/qwen3.5-35b-a3b/w8a8-base-b4096-kv3) |
| qwen3-30b-a3b-instruct-2507 | W8A8 | 6.25 | 2048 | FULL | 0 | 关闭 | [w8a8-full-kv6p25](/home/qwen-models/reports/qwen3-30b-a3b-instruct-2507/w8a8-full-kv6p25) |
| qwen3.5-9b | W8A8 | 3 | 4096 | FULL_DECODE_ONLY | 0 | 关闭 | [redhat-w8a8-base-b4096-kv3](/home/qwen-models/reports/qwen3.5-9b/redhat-w8a8-base-b4096-kv3) |
| qwen3-30b-a3b-instruct-2507-w4a8 | W4A8 | 6.25 | 2048 | FULL | 0 | 关闭 | [w4a8-full-kv6p25](/home/qwen-models/reports/qwen3-30b-a3b-instruct-2507/w4a8-full-kv6p25) |
| qwen3.6-35b-a3b-mtp2 | W8A8 | 3.75 | 8192 | FULL_DECODE_ONLY | 2 | 开启 | [w8a8-mtp2-overlap-b8192-kv3p75](/home/qwen-models/reports/qwen3.6-35b-a3b/w8a8-mtp2-overlap-b8192-kv3p75) |
| qwen3.5-35b-a3b-mtp2 | W8A8 | 3.75 | 8192 | FULL_DECODE_ONLY | 2 | 开启 | [w8a8-mtp2-overlap-b8192-kv3p75](/home/qwen-models/reports/qwen3.5-35b-a3b/w8a8-mtp2-overlap-b8192-kv3p75) |

优化包含：量化权重和INT8激活；仅加载语言模型，省去视觉塔；按模型选FULL或FULL_DECODE_ONLY、异步调度、NPUGraphEx及实际图捕获验证；混合GDN模型调整预填充预算以避免2048对齐下的长请求近似串行；固定KV预算并核对真实容量；前缀缓存align；实测共享专家流重叠及0/1/2步MTP。没有收益或出现能力问题的配置不作为默认。

`dtype=bfloat16`用于未量化层、残差、KV及部分MTP辅助层，不表示加载完整BF16模型。CANN/插件的INT4权重接口已在本机W4A8模型上实测；[官方Ascend C类型表](https://www.hiascend.com/doc_center/source/zh/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0614.html)也列出A2的int4b_t矩阵支持。W4A8并不表示全程INT4计算或两倍速度；本权重磁盘以INT8容器保存低位整数，运行时打包INT4，磁盘不会减半。

35B回补[上游PR #14618](https://github.com/vllm-project/vllm-ascend/pull/14618)，统一language_model_only预热与真实请求的输入路径，修复首次prefill的NoneType.size崩溃。RedHatAI 9B的Transformers5 tokenizer元数据通过数据盘独立兼容目录适配，原始BPE、模板及下载文件保持不变；6组token IDs与聊天模板对照通过。记录为 `/home/qwen-models/reports/redhat-tokenizer-compatibility.json`。

所有配置保持8192输入、8192输出上限、4活动请求；长输入实测输出128。worker完整16384-token并发容量属于运行时估算，没有当作四路各8192输出的实测。

## 复现与运行

```bash
source scripts/env_ascend_291.sh
systemctl stop qwen-llm.service
bash scripts/run_model.sh qwen3.6-35b-a3b
# 可选短输入加速：
bash scripts/run_model.sh qwen3.6-35b-a3b-mtp2
# 30B低显存选择：
bash scripts/run_model.sh qwen3-30b-a3b-instruct-2507-w4a8
```

每次只运行一个LLM实例。基准需先停止LLM与ASR，使用 `scripts/benchmark_quantized_model.sh MODEL_NAME MODEL_PATH FRESH_OUTPUT_DIR [serve参数]`。最后用 `poetry run python scripts/summarize_quantized_reports.py` 生成 `/home/qwen-models/reports/summary.json`。全部原始SSE、TTFT、能力答案、设备进程盘点、参数、服务日志和阶段退出码均保留。

## 全部配置记录

| 原始目录 | 能力题 | TPS1 | TPS4 | HBM/GiB | 8K重复TPS | 检索重复/混合 | 退出码 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| [qwen3-30b-a3b-instruct-2507/w4a8-full-kv6p25](/home/qwen-models/reports/qwen3-30b-a3b-instruct-2507/w4a8-full-kv6p25) | 71/72 | 75.36 | 248.75 | 28.27 | 179.36 | 0/4 / 4/4 | 1 |
| [qwen3-30b-a3b-instruct-2507/w8a8-eager-probe](/home/qwen-models/reports/qwen3-30b-a3b-instruct-2507/w8a8-eager-probe) | — | — | — | — | — | 0/4 / — | 1 |
| [qwen3-30b-a3b-instruct-2507/w8a8-final-quality-kv6p25](/home/qwen-models/reports/qwen3-30b-a3b-instruct-2507/w8a8-final-quality-kv6p25) | 72/72 | — | — | — | — | — / — | 0 |
| [qwen3-30b-a3b-instruct-2507/w8a8-full](/home/qwen-models/reports/qwen3-30b-a3b-instruct-2507/w8a8-full) | 72/72 | 78.41 | 287.37 | 43.81 | 194.07 | 0/4 / — | 1 |
| [qwen3-30b-a3b-instruct-2507/w8a8-full-kv6p25](/home/qwen-models/reports/qwen3-30b-a3b-instruct-2507/w8a8-full-kv6p25) | — | 67.24 | 270.00 | 39.81 | 192.87 | 0/4 / 4/4 | 1 |
| [qwen3.5-35b-a3b/w8a8-base-b4096-kv3](/home/qwen-models/reports/qwen3.5-35b-a3b/w8a8-base-b4096-kv3) | 72/72 | 62.62 | 184.73 | 42.18 | 106.69 | 4/4 / 4/4 | 0 |
| [qwen3.5-35b-a3b/w8a8-base-b4096-kv3-steady](/home/qwen-models/reports/qwen3.5-35b-a3b/w8a8-base-b4096-kv3-steady) | — | — | — | 42.18 | 137.30 | — / — | 0 |
| [qwen3.5-35b-a3b/w8a8-mtp1-overlap-b4096-kv3p75](/home/qwen-models/reports/qwen3.5-35b-a3b/w8a8-mtp1-overlap-b4096-kv3p75) | 71/72 | 61.12 | 181.46 | 45.59 | 70.19 | 4/4 / 4/4 | 0 |
| [qwen3.5-35b-a3b/w8a8-mtp2-overlap-b8192-kv3p75](/home/qwen-models/reports/qwen3.5-35b-a3b/w8a8-mtp2-overlap-b8192-kv3p75) | 71/72 | 74.68 | 213.15 | 46.22 | 103.81 | 4/4 / 4/4 | 0 |
| [qwen3.5-9b/redhat-w8a8-base-b4096-kv3](/home/qwen-models/reports/qwen3.5-9b/redhat-w8a8-base-b4096-kv3) | 71/72 | 75.50 | 243.38 | 19.38 | 167.98 | 4/4 / 4/4 | 0 |
| [qwen3.5-9b/redhat-w8a8-base-b4096-kv8p5](/home/qwen-models/reports/qwen3.5-9b/redhat-w8a8-base-b4096-kv8p5) | — | — | — | — | — | — / — | 1 |
| [qwen3.5-9b/redhat-w8a8-compat-b4096-kv8p5](/home/qwen-models/reports/qwen3.5-9b/redhat-w8a8-compat-b4096-kv8p5) | 71/72 | 74.67 | 241.32 | 24.87 | 177.26 | 4/4 / 4/4 | 0 |
| [qwen3.5-9b/redhat-w8a8-mtp1-b4096-kv4p5](/home/qwen-models/reports/qwen3.5-9b/redhat-w8a8-mtp1-b4096-kv4p5) | 71/72 | 70.79 | 225.27 | 23.40 | 173.29 | 4/4 / 4/4 | 0 |
| [qwen3.5-9b/redhat-w8a8-mtp2-b4096-kv5p5](/home/qwen-models/reports/qwen3.5-9b/redhat-w8a8-mtp2-b4096-kv5p5) | 70/72 | 87.31 | 265.11 | 24.32 | 166.42 | 4/4 / 4/4 | 0 |
| [qwen3.5-9b/sogagaga-w8a8-base-b4096-kv8p5](/home/qwen-models/reports/qwen3.5-9b/sogagaga-w8a8-base-b4096-kv8p5) | 71/72 | 74.32 | 242.24 | 24.87 | 169.72 | 4/4 / 4/4 | 0 |
| [qwen3.6-35b-a3b/w8a8-base](/home/qwen-models/reports/qwen3.6-35b-a3b/w8a8-base) | — | — | — | — | — | — / — | 5 |
| [qwen3.6-35b-a3b/w8a8-base-b4096-kv2p5](/home/qwen-models/reports/qwen3.6-35b-a3b/w8a8-base-b4096-kv2p5) | 71/72 | 62.30 | 186.22 | 41.70 | 68.41 | 4/4 / 4/4 | 0 |
| [qwen3.6-35b-a3b/w8a8-base-fixed](/home/qwen-models/reports/qwen3.6-35b-a3b/w8a8-base-fixed) | 71/72 | 62.02 | 186.19 | 44.90 | 59.82 | 4/4 / 4/4 | 0 |
| [qwen3.6-35b-a3b/w8a8-mtp1-b4096-kv3p75](/home/qwen-models/reports/qwen3.6-35b-a3b/w8a8-mtp1-b4096-kv3p75) | 71/72 | 60.27 | 178.92 | 45.50 | 70.10 | 4/4 / 4/4 | 0 |
| [qwen3.6-35b-a3b/w8a8-mtp2-kv3](/home/qwen-models/reports/qwen3.6-35b-a3b/w8a8-mtp2-kv3) | — | — | — | — | — | — / — | 1 |
| [qwen3.6-35b-a3b/w8a8-mtp2-overlap-b8192-kv3p75](/home/qwen-models/reports/qwen3.6-35b-a3b/w8a8-mtp2-overlap-b8192-kv3p75) | 71/72 | 71.83 | 209.78 | 46.22 | 104.19 | 4/4 / 4/4 | 0 |
| [qwen3.6-35b-a3b/w8a8-overlap-b4096-kv4](/home/qwen-models/reports/qwen3.6-35b-a3b/w8a8-overlap-b4096-kv4) | 71/72 | 63.13 | 188.25 | 43.31 | 138.83 | 4/4 / 4/4 | 0 |

初次35B首次请求崩溃、未适配的9B tokenizer、MTP2/KV3仅3.4路容量而被启动拒绝等记录都保留。退出码非零可能对应启动错误或指定检索压力项失败，不能将空TPS视为性能成绩。早期4轮长输入结果包含缓存过渡，最终默认3.5-35B的稳态结果单独补测7轮。

另一份sogagaga 9B的来源元数据提到Base、说明有歧义；本机结果与RedHatAI接近，均71/72且两组8K检索通过。保留两份，默认选用来源说明更明确的RedHatAI。

退出码0表示阶段正常执行；能力判定请同时查看逐题得分。新模型TPS均为独占结果，原有14B与ASR的共驻检查见环境升级报告。


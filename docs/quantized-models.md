# 四款 Qwen 量化模型适配与验证

状态：优化对照仍在执行。下列能力结果为已完成的阶段性验证，最终配置和汇总表将在测试结束后更新。

本次权重全部从 ModelScope 下载到数据盘 `/home/qwen-models/weights/`，项目 `model-bin/` 使用符号链接；模型、CANN、新 Poetry 环境与构建缓存均不放在系统盘。六份下载均已核对文件大小和 SHA256，固定文件 revision 与哈希保存在每个模型的 `.modelscope-manifest.json`，可审阅摘要见 [下载记录](model-downloads.json)。权重及未通过的测试结果均保留。

| 模型 | ModelScope 来源 | 磁盘 GiB | 格式 |
| --- | --- | ---: | --- |
| Qwen3.6-35B-A3B | [Eco-Tech](https://modelscope.cn/models/Eco-Tech/Qwen3.6-35B-A3B-w8a8) | 37.077 | msmodelslim W8A8_DYNAMIC |
| Qwen3.5-35B-A3B | [Eco-Tech，含 MTP](https://modelscope.cn/models/Eco-Tech/Qwen3.5-35B-A3B-w8a8-mtp) | 37.074 | msmodelslim W8A8_DYNAMIC |
| Qwen3-30B-A3B-Instruct-2507 | [vllm-ascend W8A8](https://modelscope.cn/models/vllm-ascend/Qwen3-30B-A3B-Instruct-2507-quantized.w8a8) | 29.091 | compressed-tensors W8A8 dynamic |
| Qwen3-30B-A3B-Instruct-2507 | [vllm-ascend W4A8](https://modelscope.cn/models/vllm-ascend/Qwen3-30B-A3B-Instruct-2507-quantized.w4a8) | 29.472 | attention W8A8，MoE W4A8 |
| Qwen3.5-9B | [sogagaga](https://modelscope.cn/models/sogagaga/Qwen3.5-9B-W8A8) | 13.068 | compressed-tensors W8A8 dynamic |
| Qwen3.5-9B | [RedHatAI](https://modelscope.cn/models/RedHatAI/Qwen3.5-9B-quantized.w8a8) | 13.064 | compressed-tensors W8A8 dynamic |

使用已通过原模型回归的 CANN 9.1.1 / torch-npu 2.9.1 / vLLM 0.18.0 源码环境，详见 [升级验收](cann911-torch291.md)。`dtype=bfloat16` 用于未量化层及 KV；没有加载完整 BF16 模型作推理对照，因此不能据此估计量化相对原权重的能力损失。

35B 权重包含视觉部分，本服务通过 `language_model_only=True` 只加载文本模型。已回补 [上游 PR #14618](https://github.com/vllm-project/vllm-ascend/pull/14618)：预热与实际请求统一使用 `supports_mm_inputs` 判断输入路径，避免首次真实请求时 `NoneType.size` 崩溃。源补丁纳入安装脚本。

官方 [Ascend C Matmul 类型表](https://www.hiascend.com/doc_center/source/zh/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0614.html)列出 Atlas A2 的 `int4b_t × int4b_t` 支持；本次 W4A8 表示 INT4 权重、INT8 激活，不等同于全程 INT4 计算或固定两倍算力。910B/A2 的 [CANN QuantMatmulV4 API](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-nn/aclnnQuantMatmulV4.md) 包含 INT4 路径。下载的 30B W4A8 将低位整数保存在 INT8 容器里，磁盘体积不会减半；只有实际装载、INT4 打包、正确性和测速完成后，才判定本机部署可用。

能力验证使用 40 道固定基础题与 GSM8K test 固定前 32 题，温度零、关闭思考，答案严格判定，并保存每题完整输出。GSM8K 来自 OpenAI 官方仓库固定提交 `3101c7d5072418e28b9008a6636bde82a006892c`，下载在 `/home/qwen-models/evaluation/`。这 72 题只用于发现明显能力问题，不代表完整基准成绩。

长输入验证包括四路各自不同的 8192-token 校验码检索，分别使用重复填充和混合中文记录。两个材料均保存原始答案；不能把某一组检索通过解释为所有长上下文任务均可靠。短输入 TPS 使用固定成语提示、128 输出 token、1/2/4 并发各三轮；8 并发验证四路容量的 429 拒绝。长输入测速为四路各 8192 输入 + 128 输出，首轮冷缓存与最后三轮重复请求分别统计；混合模型保留额外过渡轮，逐请求记录缓存命中，未命中不能算作热缓存。TPS 是实际成功输出 token 数除以整个并发批次 makespan。

原始结果在 `/home/qwen-models/reports/`，包括 doctor、health、完整能力题、检索输出、SSE/TTFT、设备进程盘点、服务日志与各阶段退出码。每次测速独占同一 NPU，ASR 暂停；失败的量化、执行模式和能力输出均保留，不用成功速度覆盖失败结论。

Qwen3.6 首次有效能力测试为 71/72，基础题 40/40、GSM8K 31/32、执行错误零；错题为树木投资回本年份与开始盈利年份的区分，原答案完整保留。此结果属于量化模型本机表现，不能归因为量化损失。原始目录 `qwen3.6-35b-a3b/w8a8-base-fixed/`。

Qwen3.6 的无 MTP 基线已通过两种四路 8192-token 材料检索，各 4/4。短/长输入测速在插件编译完成后执行，期间无其他 NPU 模型进程。

Qwen3.6 的 MTP=2 / KV=3GiB 试验经真实 worker RPC 核验，仅支持约 3.40 路完整 16384-token 请求，启动时主动拒绝提供服务；日志位于 `qwen3.6-35b-a3b/w8a8-mtp2-kv3/`。它属于容量不足的配置试验，不作为能力或 TPS 结果。新启动器同时验证真实图捕获计数与缓存容量。

RedHatAI 9B 的 tokenizer 配置来自 Transformers 5，声明 `TokenizersBackend`。服务在数据盘 `tokenizer-adapters/` 创建内容寻址的兼容目录，改用 Qwen2TokenizerFast 包装原始 BPE，并引用原始聊天模板；模型下载目录保持不变。六组文字/特殊标记/8192-token 输入与原 BPE token IDs 一致，聊天模板 token IDs 对照通过。记录为 `/home/qwen-models/reports/redhat-tokenizer-compatibility.json`。

30B W4A8 首轮严格能力得分 71/72，唯一未通过项的数值答案正确，但使用 `**FINAL: 60**`，违反指定的裸行格式，仍按既定规则记未通过；未据此声称数学能力损失。混合中文 8K 检索 4/4，重复 `data` 压力材料 0/4，失败输出完整保留。

## 首轮已完成的独占测试

这些是调优前/阶段性配置，最终选择会依据后续对照更新。短 TPS 为三轮中位数，整卡 HBM 是轮次采样最大值，非连续峰值。

| 模型与配置 | 严格能力题 | 重复/混合 8K 检索 | TPS 1路 | TPS 2路合计 | TPS 4路合计 | 整卡 HBM GiB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.6-35B W8A8，batch2048/KV6，无MTP | 71/72 | 4/4、4/4 | 62.02 | 99.17 | 186.19 | 44.90 |
| Qwen3.5-35B W8A8，batch4096/KV3，无MTP | 72/72 | 4/4、4/4 | 62.62 | 100.66 | 184.73 | 42.18 |
| 30B W8A8，FULL/KV10.25 | 72/72 | 0/4、待补混合材料 | 78.41 | 148.96 | 287.37 | 43.81 |
| 30B W4A8，FULL/KV6.25 | 71/72（唯一扣分为格式） | 0/4、4/4 | 75.36 | 141.92 | 248.75 | 28.27 |
| 9B RedHatAI W8A8，batch4096/KV8.5 | 71/72 | 4/4、4/4 | 74.67 | 128.88 | 241.32 | 24.87 |

9B 的唯一未通过题为 GSM8K 盈亏平衡年份题，回答反复讨论并在2048输出token预算内未给出指定的最终答案。30B W8A8在eager下也未通过重复填充材料，不能只归因为图执行；未运行完整BF16基线，无法判定属于原模型还是量化/后端的损失。各份权重和失败输出均保留。

所有配置保持输入8192、输出上限8192、4个活动请求。长输入实测为每路8192输入加128输出；worker返回的完整16384-token并发容量是运行时估算，未将它当作四路各8192输出的实测结果。

9B 两步 MTP 的严格得分降至70/72，基础算术题 `29×31` 返回 `897`，正确答案为 `899`；无MTP基线此题通过。两组8K检索仍各4/4。该配置不设为默认，完整错误输出与参数保存在 `qwen3.5-9b/redhat-w8a8-mtp2-b4096-kv5p5/`，原权重继续保留。

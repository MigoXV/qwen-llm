# 已有模型能力成绩

2026-10-16历史记录。仅查阅当时已有报告和权重发布者说明，没有新增模型评测。当前容量以`configs/models.json`及[热点报告](qwen38-hotspots.md)为准；原生INT8激活实验及其别名现已迁入`tests/performance/qwen38/`，不属于生产入口。下列配置和成绩保留当时口径。
服务默认选择 `qwen3.8-27b-mtp2`：两步MTP、8192 batch tokens、9.375 GiB KV、
FULL_DECODE_ONLY、异步调度、NPUGraphEx、前缀缓存，`upstream` W4线性后端。
8K输入、8K输出上限和4路活动请求保持原有配置。

## 本机已有结果

| 模型和实际执行方式 | 基础题 | GSM8K前32题 | 合计 | 8K检索 |
| --- | ---: | ---: | ---: | --- |
| Qwen3-14B，W8A8 | 40/40 | 未测 | 40/40 | 四路不同材料4/4 |
| Qwen3.6-35B-A3B，W8A8 | 40/40 | 31/32 | 71/72（98.61%） | 重复材料4/4、混合中文4/4 |
| Qwen3.8-27B，当前默认MTP2，FFN为W4/BF16激活 | 40/40 | 31/32 | 71/72（98.61%） | 重复材料4/4、混合中文4/4 |
| Qwen3.8-27B，原生INT8激活W4A8实验 | 39/40 | 31/32 | 70/72（97.22%） | 重复材料4/4、混合中文4/4 |

14B和3.6-35B在本机使用的是 **W8A8**。未查到与本机对应的这两个模型的
**W4A8能力报告**，不能把上表的W8A8成绩当作W4A8成绩。
27B权重目录虽然命名为W4A8，默认FFN算子实际使用INT4权重和BF16激活；
只有显式 `qwen3.8-27b-w4a8` 配置使用原生INT8激活。

27B默认MTP2唯一失分为GSM8K的盈利年份题：回答第12年回本，标准答案要求第13年开始净盈利。
原生INT8激活实验另将 `29×31` 答为900，正确答案899；两个配置均保留。
较大KV预算10.5 GiB的MTP2历史配置得分72/72，未据此推断增加缓存会提高能力。

这些本机结果是温度0、关闭思考的小样本检查，14B也没有运行同一套32题数学子集。
没有对应的BF16完整对照，不能从这些成绩给出量化能力损失百分比或完整能力排名。

原始逐题回答：

- [14B W8A8](/home/qwen-upgrade-20261015/final-verification/candidate/llm-quality.json)
- [3.6-35B W8A8](/home/qwen-models/reports/qwen3.6-35b-a3b/w8a8-overlap-b4096-kv4/quality.json)
- [27B默认MTP2](/home/qwen-models/reports/qwen3.8-27b/w4a8-mtp2-b8192-kv9p375/quality.json)
- [27B原生INT8激活](/home/qwen-models/reports/qwen3.8-27b/w4a8-gmm-b4096-kv6p5/quality.json)

## 查到的公开成绩

本机3.6-35B所用的[Eco-Tech W8A8模型卡](https://www.modelscope.cn/models/Eco-Tech/Qwen3.6-35B-A3B-w8a8/summary)
报告 **GPQA 83.232%，本地浮点对照82.997%**，相差+0.235个百分点。
发布者使用Atlas 800T A3、vLLM-Ascend v0.18.0rc1、256000上下文，说明成绩存在波动。
这份报告未显示该项能力下降；它属于发布者环境，不能视作本机8K配置的GPQA实测。
下载时保存的[模型卡原文](/home/qwen-models/weights/Qwen3.6-35B-A3B-W8A8/README.md)亦含该表。

本机[14B发布者模型卡](/home/qwen-llm-build/models/Qwen3-14B-W8A8-Ascend/README.md)
未提供MMLU、GPQA或完整GSM8K成绩。
本机27B是自行量化，没有找到针对这一份权重的公开完整基准报告。

另有其他发布者的[27B CUDA W4A8模型卡](https://huggingface.co/schoggie/Qwen3.8-27B-W4A8)
自报HumanEval pass@1为92.1%，其AWQ对照93.3%；256K检索均20/20。
其权重、量化方法、运行后端与本机版本不同，这些数字仅作同名模型其他量化版本的参考。

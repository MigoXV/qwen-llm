# Qwen3.8-27B 自制 W4A8：量化、适配与本机验证

2026-10-16，Asia/Shanghai。原始权重已完成校验与自量化，模型已接入本项目的
V1 AsyncLLM 无状态 Responses 服务。当前通过完整验收的图配置：单路31.73 TPS、
四路合计97.68 TPS，整卡HBM采样最大33.20 GiB。原生INT8激活后端已测完，
速度与能力均未超过默认路径，保留为独立实验。可选MTP两步配置单路47.87 TPS、
四路合计130.00 TPS，HBM约40.25 GiB，适合短输入加速；按用户选择，现已设为默认启动配置；低显存配置仍可显式选择。

## 环境与权重

Ascend910B4-1，物理4/逻辑0，64 GiB；驱动26.1.1，CANN 9.1.1，
Python3.10.22、torch/torch-npu2.9.1、源码vLLM0.18.0、vLLM-Ascend0.18.0、Transformers4.57.6。
模型是64层Dense混合GDN/完整注意力架构；此处单卡、TP=1、只测文本。

| 内容 | 数据盘目录 | 文件大小/GiB |
| --- | --- | ---: |
| 原始BF16 | `/home/model-bin/Qwen/Qwen3.8-27B` | 51.769 |
| 自量化部署权重 | `/home/model-bin/Qwen/Qwen3.8-27B-W4A8-Ascend` | 24.077 |
| 早期group256候选 | `/home/model-bin/Qwen/Qwen3.8-27B-W4A8-Ascend-group256-nd-only` | 23.330 |

原始模型来源为[Qwen/Qwen3.8-27B](https://modelscope.cn/models/Qwen/Qwen3.8-27B)，
全部18个分片及推理所需文件逐项核对ModelScope文件revision、大小和SHA256。
`.gitattributes`没有下载，不属于推理必需文件。没有使用第三方量化权重代替本次自量化。
模型、工具依赖、缓存和原始报告均放在`/home`数据盘；原始权重未覆盖。

原始清单：`/home/qwen-quantize-20261016/reports/source-manifest.json`。
导出清单：部署权重目录中的`quantization_manifest.json`，含各分片SHA256和逐层诊断。

## 自量化方案

固定使用[Ascend/msmodelslim](https://github.com/Ascend/msmodelslim)
提交`9e19c1bcfe217b63666921606fc0abc24bf0709a`，逐张量CPU量化，16线程，
不向NPU加载完整BF16模型。额外依赖位于`/home/qwen-quantize-20261016/dependencies`，
没有升级当前Poetry推理环境。完整量化约354秒，布局适配另计。

- 192个FFN gate/up/down矩阵：对称分阶段INT8→group256 INT4，输出维双nibble打包。
- 160个完整注意力及GDN qkv/z投影：按输出通道INT8权重、动态W8A8。
- GDN out_proj、in_proj_a/b、递归控制、归一化、embedding/lm_head、视觉和MTP权重保留浮点。
- 配置、tokenizer与聊天模板沿用原始文件；只加载语言模型，MTP仅在显式启用时使用。

这是无校准数据的PTQ，没有运行GPTQ、AWQ、SmoothQuant或量化训练。
权重RMSE不能当作模型能力损失；下文的72题仅为能力冒烟，没有同模型BF16完整精度对照。

早期group256权重通过了ND算子检查，但整模型使用NZ布局时被CANN拒绝：
`npu_weight_quant_batchmatmul`的NZ INT4要求group64/128。
因此把每个256组的尺度等值拆成两个128组，整数权重保持不变，形成24.077 GiB版本。
全量核对2,095个其他张量逐值不变，384个尺度/偏移张量按预期等值细分。
记录为`/home/qwen-quantize-20261016/reports/lossless-group-subdivision.json`。
早期权重及失败日志保留；没有将失败的ND候选当作部署版本。

## 两条执行路径

| 后端选项 | FFN执行路径 | 激活 | 输出 |
| --- | --- | --- | --- |
| `upstream` | 上游`npu_weight_quant_batchmatmul` | BF16 | BF16 |
| `gmm` | `npu_dynamic_quant`→单专家`npu_grouped_matmul`，权重保持INT4打包 | INT8、逐token动态尺度 | BF16 |

旧插件的`W4A8_DYNAMIC`名称不代表其Dense默认实现使用INT8激活。
本项目显式提供`--w4a8-linear-backend gmm`，沿用上游MoE的尺度与补偿偏置约定，
让Dense FFN实际使用INT8激活接口。模型仍是Dense，只复用单专家GMM算子。

GMM保留FP32导入尺度，再按算子约定编码；等值的相邻128组可合并回256组，
无须重新量化。GMM与默认反量化算子的分组限制不同。
此后端仅验证对称ModelSlim版本1.0.0、TP=1、BF16残差；非法dtype在创建Worker前拒绝。
Worker就绪信息报告实际使用的线性层数量和分组，避免只检查请求参数。

## 数值与能力检查

线性层门槛在测试前固定为：相对对应量化参考的L2误差低于1%。
这些检查不包含相对原始BF16模型的量化损失。

| 算子检查 | 形状与batch | 结果 |
| --- | --- | --- |
| INT4存储 | 有符号边界与原生打包 | 逐值一致 |
| 上游NZ路径 | gate `[17408,5120]`、down `[5120,17408]`；1/4/128 | L2约0.17%～0.21% |
| 原生W4A8 | 上述两种与融合gate/up `[34816,5120]`；1/4/128/4096，共12项 | 最大L2为0.213% |

原生W4A8参考先对激活执行相同动态INT8量化，再使用反量化权重做FP32线性计算，
分离算子实现误差与激活量化误差。对应记录：

- `/home/qwen-quantize-20261016/reports/full-size-linear-nz128-conformance.json`
- `/home/qwen-quantize-20261016/reports/native-full-size-conformance.json`

整模型使用40道基础题、GSM8K test前32题、温度0、关闭思考。
GSM8K固定数据SHA256为`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`。
两类8K检索均为四路不同校验码，覆盖重复data填充与混合中文材料。

| 配置 | 基础题 | GSM8K | 执行错误 | 重复8K | 混合中文8K |
| --- | ---: | ---: | ---: | ---: | ---: |
| 上游W4 Eager | 40/40 | 31/32 | 0 | 4/4 | 4/4 |
| 上游W4图执行 | 40/40 | 31/32 | 0 | 4/4 | 4/4 |
| 原生W4A8图执行 | 39/40 | 31/32 | 0 | 4/4 | 4/4 |
| MTP2 / KV10.5 GiB | 40/40 | 32/32 | 0 | 4/4 | 4/4 |
| MTP2 / KV9.375 GiB | 40/40 | 31/32 | 0 | 4/4 | 4/4 |

两条路径的72项判定全部一致，51项完整输出文本一致；不宣称逐token完全等价。
唯一失分为GSM8K第13题：输出12年回本，标准要求第13年开始净盈利；保持原判。
Eager的32道数学题全部正常完成，共11574输出tokens，单题最长665 tokens，没有预算截断。
没有同模型BF16完整基线，因此不能把该失分直接归因为量化。

原生W4A8另将`29×31`答为900，正确答案899，没有达到预先设定的基础题40/40门槛。
原生算子数值检查通过不等于端到端能力没有变化。因此保留该路径，未设为默认，
别名为`qwen3.8-27b-w4a8`；`qwen3.8-27b`及默认MTP2配置的FFN使用INT4权重与BF16激活。
MTP2小缓存配置的唯一失分仍为同一盈利年份题；大缓存配置该题答对。
这类小样本结果不用于推断能力全面提高，也没有把答案差异归因为缓存预算。

## 性能与内存

同一HTTP/SSE客户端、同一提示、38输入+128输出tokens，1/2/4路各三轮取中位数，
各轮实际输出均达到128。输出TPS=成功生成token总数/整轮makespan，四路列为合计吞吐。
前缀缓存开关开启，但短输入实际命中为0。另以8路验证四路之外的429拒绝，
每轮4成功、4拒绝；不把8路当作模型并发容量。

| 配置 | 1路TPS | 2路TPS | 4路TPS | 整卡HBM/GiB |
| --- | ---: | ---: | ---: | ---: |
| 上游W4 Eager | 5.61 | 10.33 | 20.42 | 32.85 |
| 上游W4图执行 | 31.73 | 54.13 | 97.68 | 33.20 |
| 原生W4A8图执行 | 27.84 | 48.93 | 93.10 | 33.36 |
| MTP2 / KV10.5 GiB | 47.35 | 75.38 | 127.65 | 41.36 |
| MTP2 / KV9.375 GiB | 47.87 | 75.24 | 130.00 | 40.25 |

默认图路径三轮范围：单路31.61～31.75、两路53.83～54.19、四路97.52～97.96 TPS。
对应首token延迟样本中位数约399/779/809 ms；小样本不能代表生产尾延迟。
单路与四路相对Eager分别提高约5.66倍、4.78倍。

四路长输入每路8192输入+128输出，7轮；第1轮冷缓存，最后3轮重复请求取中位数。

| 配置 | 冷缓存首轮TPS | 重复请求TPS | 每路重复命中tokens |
| --- | ---: | ---: | ---: |
| 上游W4图执行 | 29.92 | 85.72 | 7680 |
| 原生W4A8图执行 | 25.95 | 81.15 | 7680 |
| MTP2 / KV10.5 GiB | 35.91 | 61.00 | 6144 / 6144 / 6144 / 0 |
| MTP2 / KV9.375 GiB | 36.02 | 49.57 | 6144 / 6144 / 0 / 0 |

默认路径重复末三轮85.55～85.82 TPS，命中模式稳定。首轮只有一个样本，与稳态统计区分。
HBM为整卡轮次采样最大值，包含约3.34 GiB驱动基础占用，不是连续瞬时峰值。

Eager、普通图与GMM的共同参数：TP=1、8192输入/8192输出上限、4活动请求、4096-token分块预填充、
6.5 GiB固定KV预算、前缀缓存align、异步调度。图执行为FULL_DECODE_ONLY、NPUGraphEx、
实际捕获1/2/4三个图。没有CPU offload，基线未启用MTP。

6 GiB缓存只能估算3.71路完整上下文，启动检查已拒绝该配置。
6.5 GiB预算实分配约6.44 GiB，worker估算完整16384-token容量4.0路。
容量估算与实际8192输入+128输出测试分别报告，没有宣称四路满8192输出已实测。

MTP组合配置使用两步推测、8192-token预填充与3/6/12捕获形状，真实drafter为
`AscendEagleProposer`。草稿权重保留BF16，不是加载整套BF16目标模型。
10.5 GiB预算估算容量4.48路；根据真实块布局缩到9.375 GiB，实分配约9.31 GiB、容量4.0路。
小缓存保留短输入收益，HBM减少约1.11 GiB，但重复8K缓存命中减少，吞吐也下降。
因此把`qwen3.8-27b-mtp2`保留为可选加速配置，没有替换33.20 GiB的默认配置。

## 复现与原始数据

```bash
source scripts/env_ascend_291.sh
poetry run qwen-llm quantize-w4a8 \
  --model-path /home/model-bin/Qwen/Qwen3.8-27B \
  --output-dir /home/model-bin/Qwen/Qwen3.8-27B-W4A8-Ascend \
  --source-manifest /home/qwen-quantize-20261016/reports/source-manifest.json \
  --modelslim-root /home/qwen-quantize-20261016/src/msmodelslim \
  --tool-dependencies /home/qwen-quantize-20261016/dependencies \
  --w4-scope mlp --group-size 256 --threads 16
# 输出目录必须为空；已有候选保留，重新导出时指定新目录。
poetry run python scripts/verify_w4_linear.py \
  --model-path /home/model-bin/Qwen/Qwen3.8-27B-W4A8-Ascend \
  --output /home/qwen-quantize-20261016/reports/recheck-upstream.json
poetry run python scripts/verify_w4_linear.py --backend gmm --prefill-tokens 4096 \
  --model-path /home/model-bin/Qwen/Qwen3.8-27B-W4A8-Ascend \
  --output /home/qwen-quantize-20261016/reports/recheck-gmm.json
bash scripts/run_model.sh qwen3.8-27b
# 短输入加速，内存和长输入缓存取舍见上表：
bash scripts/run_model.sh qwen3.8-27b-mtp2
# 保留的原生INT8激活实验，有已知算术错误：
bash scripts/run_model.sh qwen3.8-27b-w4a8
```

组件检查和服务均占用NPU，应在独占设备下分别运行。导出期间带有
`.quantization.incomplete`标记，启动诊断拒绝加载中间权重。

停止已有实例后，可在新目录复测默认配置；脚本负责启动、测试和关闭：

```bash
bash scripts/benchmark_quantized_model.sh qwen3.8-27b \
  /home/model-bin/Qwen/Qwen3.8-27B-W4A8-Ascend \
  /home/qwen-models/reports/qwen3.8-27b/recheck-default \
  --runner aclgraph --graph-mode FULL_DECODE_ONLY \
  --batch-tokens 4096 --kv-cache-gib 6.5 --async-scheduling \
  --npugraph-ex --prefix-caching --w4a8-linear-backend upstream
```

MTP复测改用新输出目录，设置`--batch-tokens 8192 --kv-cache-gib 9.375 --speculative-tokens 2`。
原生W4A8实验设置`--w4a8-linear-backend gmm`，其余沿用普通图参数。

原始目录在`/home/qwen-models/reports/qwen3.8-27b/`，包含全部回答、SSE终态、
TTFT、进程盘点、实际参数、服务日志和阶段退出码；失败配置同样保留。

| 目录 | 结果 |
| --- | --- |
| `w4a8-eager-b4096-kv6` | 早期group256 NZ启动失败 |
| `w4a8-eager-b4096-kv6-g128` | group128已兼容，KV容量不足，拒绝启动 |
| `w4a8-eager-b4096-kv6p5-g128` | Eager完整能力、检索、短输入测速通过 |
| `w4a8-b4096-kv6p5-g128` | 图执行完整能力、检索、短/长输入测试通过 |
| `w4a8-gmm-b4096-kv6p5` | 原生W4A8，70/72，有新增算术错误且较慢，保留未默认 |
| `w4a8-mtp2-b8192-kv10p5` | 72/72，MTP两步，较大缓存对照 |
| `w4a8-mtp2-b8192-kv9p375` | 71/72，四路容量4.0，低缓存MTP可选配置 |

版本化数值、参数和SHA256索引见[qwen38-w4a8-results.json](qwen38-w4a8-results.json)。
数据盘汇总：`/home/qwen-models/reports/qwen3.8-27b-summary.json`。
Eager/图判定对齐：该模型报告根目录的`eager-graph-quality-alignment.json`。
阶段退出码0只表示请求和测量完成，不表示所有能力答案正确；判定以逐题记录为准。

自动化测试67项通过，Poetry配置检查通过；仅扫描Poetry运行时的pip依赖检查通过。
启用CANN的PYTHONPATH后，全量pip检查也会扫描工具包自带的可选分析工具，
报告argparse/plotly等缺依赖及ms-service-profiler的OpenTelemetry版本约束冲突。
本轮没有为这些可选工具改动推理环境；两种检查分别保留于
`/home/qwen-quantize-20261016/reports/pip-check-runtime.log`和`pip-check.log`。

从配置入口`run_model.sh qwen3.8-27b`重新启动后，OpenAI SDK3.31严格响应校验下的
非流式与SSE均通过，8193-token输出请求返回400。
记录为`/home/qwen-models/reports/qwen3.8-27b/profile-http-smoke/sdk-smoke.json`。
SIGTERM正常退出已核验，测试后NPU无残留拥有进程、8100端口关闭，LLM与ASR系统服务均保持inactive；
见同一报告根目录的`lifecycle.json`。

官方[Qwen3.8-27B教程](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.8-27B.html)
从vLLM-Ascend0.23开始验证此模型。本机复用其`qwen3_5`架构的0.18实现，
兼容性以本次实机结果为准，不宣称属于上游0.18的正式支持矩阵。

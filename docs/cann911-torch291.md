# CANN 9.1.1 / Torch 2.9.1 最终升级验收

2026-10-15，Ascend910B4-1（物理4、逻辑0，64 GiB），驱动26.1.1保持不变。两个项目保留新环境，原环境与CANN仍保留。当前最终插件构建下重新完成同负载测试：**83项指标均未触发20%回退门槛**。

| 组件 | LLM旧环境 | ASR旧环境 | 两项目新环境 |
| --- | --- | --- | --- |
| Python | 3.10.22 | 3.10.22 | 3.10.22 |
| Torch / torch-npu | 2.7.1 / 2.7.1 | 2.8.0 / 2.8.0 | 2.9.1 / 2.9.1 |
| vLLM | 0.11.0+empty | 0.13.0+empty | 0.18.0+empty |
| vLLM Ascend | 0.11.0rc1 | 0.13.0rc1 | 0.18.0+torch291.cann911 |
| CANN | 8.3.RC1 | 8.3.RC1 | 9.1.1 |
| Transformers | 4.57.1 | 4.57.6 | 4.57.6 |

Torch运行时标识为2.9.1+cpu，实际使用torch-npu在NPU计算。CANN Toolkit、910B ops、NNAL以及两个独立Poetry环境均安装于 `/home/qwen-upgrade-20261015/`，未覆盖旧CANN或更换驱动。进程运行库已核对来自新CANN；新源码版本、补丁、约束和最终wheel哈希见 [deploy/ascend-291](../deploy/ascend-291/)。这是本机实测组合，不宣称属于上游原版安装矩阵。

Triton-Ascend为3.2.2；按包依赖先安装基础Triton3.5.0，再覆盖Ascend实现。两环境pip check与Poetry check通过。LLM58项测试通过；ASR36项常规测试通过、7项设备测试默认跳过，迁移期间另有4项真实NPU注意力/RoPE数值对齐测试通过。

## 14B LLM

沿用原Qwen3-14B W8A8权重，8192输入、8192输出上限、4活动请求、2048批处理token、10.25GiB固定KV。最终使用FULL图，捕获[1,2,4,64,128,256]，异步调度、NPUGraphEx和前缀缓存开启。设备profile预算比例0.82，实际KV仍按显式字节预算分配。worker核验图已捕获、完整上下文容量约4.09路。

最终固定答案40/40、四路不同8192-token材料检索4/4通过。自由文本没有要求逐字重现旧版本；测试未把四路8K输入+128输出视为四路满8K输出的实测。

| 并发 | 旧TPS | 新TPS | 吞吐变化 |
| ---: | ---: | ---: | ---: |
| 1 | 46.39 | 53.25 | +14.79% |
| 2 | 87.12 | 103.78 | +19.12% |
| 4 | 170.63 | 202.26 | +18.54% |

短输入各并发三轮、每次128输出token。8并发验证429过载，不作为吞吐退化门槛。长输入按至少三轮缓存命中结果比较；首轮冷缓存另存。

初期PIECEWISE配置使吞吐下降约31%～33%；FULL_DECODE_ONLY改善吞吐但短输入首token更慢。关闭异步或NPUGraphEx未解决首token问题，最终增加短输入FULL图捕获后同时改善吞吐与首token。失败配置与中断的noex冷缓存记录均保留；中断记录不用于验收。

## ASR

原Qwen3-ASR-1.7B，BF16、8执行槽、32RPC容量、1024上下文与批处理token、896MiB固定KV、关闭前缀缓存，异步调度、FULL_DECODE_ONLY和ASR融合RoPE开启。24组负载各三轮，816次请求全部成功，816条转写与旧环境逐字一致。最差吞吐下降13.94%，最差P95首字/总延迟增加16.80%；HBM采样最大值10509→9772MiB。详细24组表格见 [ASR报告](/root/repositories/qwen-asr/docs/cann911-torch291.md)。

## 逐项指标

吞吐下降超过20%或P95延迟增加超过20%的组需要额外三轮确认，确认后回退。最终83项无超标组。负值代表改善。

| 项目 | 负载 | 指标 | 旧环境 | 新环境 | 退化幅度 | 超过20% |
| --- | --- | --- | ---: | ---: | ---: | --- |
| llm | short/concurrency=1 | tokens/s | 46.3921 | 53.2548 | -14.8% | 否 |
| llm | short/concurrency=1 | seconds_p95 | 2.7646 | 2.4118 | -12.8% | 否 |
| llm | short/concurrency=1 | ttft_seconds_p95 | 0.0634 | 0.0446 | -29.7% | 否 |
| llm | short/concurrency=2 | tokens/s | 87.1246 | 103.7837 | -19.1% | 否 |
| llm | short/concurrency=2 | seconds_p95 | 2.9924 | 2.4778 | -17.2% | 否 |
| llm | short/concurrency=2 | ttft_seconds_p95 | 0.1416 | 0.0689 | -51.3% | 否 |
| llm | short/concurrency=4 | tokens/s | 170.6291 | 202.2594 | -18.5% | 否 |
| llm | short/concurrency=4 | seconds_p95 | 3.0449 | 2.5352 | -16.7% | 否 |
| llm | short/concurrency=4 | ttft_seconds_p95 | 0.1445 | 0.0752 | -48.0% | 否 |
| llm | 8192-input/hot/concurrency=4 | tokens/s | 123.3759 | 141.5350 | -14.7% | 否 |
| llm | 8192-input/hot/concurrency=4 | ttft_seconds_p95 | 0.2391 | 0.1980 | -17.2% | 否 |
| asr | 5.130s/clients=1 | throughput_x_median | 21.6816 | 19.4975 | 10.1% | 否 |
| asr | 5.130s/clients=1 | first_text_s_p95 | 0.1280 | 0.1367 | 6.8% | 否 |
| asr | 5.130s/clients=1 | latency_s_p95 | 0.2381 | 0.2647 | 11.1% | 否 |
| asr | 5.130s/clients=2 | throughput_x_median | 32.1161 | 29.4833 | 8.2% | 否 |
| asr | 5.130s/clients=2 | first_text_s_p95 | 0.2140 | 0.2254 | 5.3% | 否 |
| asr | 5.130s/clients=2 | latency_s_p95 | 0.3321 | 0.3586 | 8.0% | 否 |
| asr | 5.130s/clients=4 | throughput_x_median | 62.4647 | 56.2091 | 10.0% | 否 |
| asr | 5.130s/clients=4 | first_text_s_p95 | 0.2258 | 0.2426 | 7.5% | 否 |
| asr | 5.130s/clients=4 | latency_s_p95 | 0.3427 | 0.3826 | 11.6% | 否 |
| asr | 5.130s/clients=8 | throughput_x_median | 118.4117 | 106.6706 | 9.9% | 否 |
| asr | 5.130s/clients=8 | first_text_s_p95 | 0.2299 | 0.2486 | 8.1% | 否 |
| asr | 5.130s/clients=8 | latency_s_p95 | 0.3478 | 0.3881 | 11.6% | 否 |
| asr | 5.130s/clients=16 | throughput_x_median | 122.0721 | 109.7597 | 10.1% | 否 |
| asr | 5.130s/clients=16 | first_text_s_p95 | 0.5535 | 0.6125 | 10.7% | 否 |
| asr | 5.130s/clients=16 | latency_s_p95 | 0.6722 | 0.7534 | 12.1% | 否 |
| asr | 5.130s/clients=32 | throughput_x_median | 122.3627 | 110.3743 | 9.8% | 否 |
| asr | 5.130s/clients=32 | first_text_s_p95 | 1.2632 | 1.3574 | 7.5% | 否 |
| asr | 5.130s/clients=32 | latency_s_p95 | 1.3844 | 1.4991 | 8.3% | 否 |
| asr | 10.000s/clients=1 | throughput_x_median | 30.2865 | 26.6465 | 12.0% | 否 |
| asr | 10.000s/clients=1 | first_text_s_p95 | 0.1300 | 0.1407 | 8.3% | 否 |
| asr | 10.000s/clients=1 | latency_s_p95 | 0.3335 | 0.3791 | 13.7% | 否 |
| asr | 10.000s/clients=2 | throughput_x_median | 48.5905 | 43.1978 | 11.1% | 否 |
| asr | 10.000s/clients=2 | first_text_s_p95 | 0.2121 | 0.2297 | 8.3% | 否 |
| asr | 10.000s/clients=2 | latency_s_p95 | 0.4274 | 0.4778 | 11.8% | 否 |
| asr | 10.000s/clients=4 | throughput_x_median | 91.8795 | 81.5526 | 11.2% | 否 |
| asr | 10.000s/clients=4 | first_text_s_p95 | 0.2321 | 0.2496 | 7.5% | 否 |
| asr | 10.000s/clients=4 | latency_s_p95 | 0.4440 | 0.4967 | 11.9% | 否 |
| asr | 10.000s/clients=8 | throughput_x_median | 172.2324 | 154.0615 | 10.6% | 否 |
| asr | 10.000s/clients=8 | first_text_s_p95 | 0.2506 | 0.2663 | 6.2% | 否 |
| asr | 10.000s/clients=8 | latency_s_p95 | 0.4705 | 0.5223 | 11.0% | 否 |
| asr | 10.000s/clients=16 | throughput_x_median | 175.0279 | 158.7671 | 9.3% | 否 |
| asr | 10.000s/clients=16 | first_text_s_p95 | 0.6924 | 0.7504 | 8.4% | 否 |
| asr | 10.000s/clients=16 | latency_s_p95 | 0.9269 | 1.0083 | 8.8% | 否 |
| asr | 10.000s/clients=32 | throughput_x_median | 157.4249 | 159.1805 | -1.1% | 否 |
| asr | 10.000s/clients=32 | first_text_s_p95 | 1.8194 | 2.0487 | 12.6% | 否 |
| asr | 10.000s/clients=32 | latency_s_p95 | 2.0356 | 2.3057 | 13.3% | 否 |
| asr | 20.000s/clients=1 | throughput_x_median | 38.0186 | 32.7194 | 13.9% | 否 |
| asr | 20.000s/clients=1 | first_text_s_p95 | 0.1429 | 0.1530 | 7.1% | 否 |
| asr | 20.000s/clients=1 | latency_s_p95 | 0.5385 | 0.6289 | 16.8% | 否 |
| asr | 20.000s/clients=2 | throughput_x_median | 64.8985 | 56.6148 | 12.8% | 否 |
| asr | 20.000s/clients=2 | first_text_s_p95 | 0.2284 | 0.2457 | 7.6% | 否 |
| asr | 20.000s/clients=2 | latency_s_p95 | 0.6441 | 0.7332 | 13.8% | 否 |
| asr | 20.000s/clients=4 | throughput_x_median | 122.1803 | 108.2921 | 11.4% | 否 |
| asr | 20.000s/clients=4 | first_text_s_p95 | 0.2503 | 0.2663 | 6.4% | 否 |
| asr | 20.000s/clients=4 | latency_s_p95 | 0.6714 | 0.7484 | 11.5% | 否 |
| asr | 20.000s/clients=8 | throughput_x_median | 184.3067 | 164.4414 | 10.8% | 否 |
| asr | 20.000s/clients=8 | first_text_s_p95 | 0.4425 | 0.4718 | 6.6% | 否 |
| asr | 20.000s/clients=8 | latency_s_p95 | 0.8673 | 0.9722 | 12.1% | 否 |
| asr | 20.000s/clients=16 | throughput_x_median | 188.4595 | 167.1183 | 11.3% | 否 |
| asr | 20.000s/clients=16 | first_text_s_p95 | 1.2672 | 1.4264 | 12.6% | 否 |
| asr | 20.000s/clients=16 | latency_s_p95 | 1.7030 | 1.9422 | 14.0% | 否 |
| asr | 20.000s/clients=32 | throughput_x_median | 194.1206 | 169.8462 | 12.5% | 否 |
| asr | 20.000s/clients=32 | first_text_s_p95 | 2.8741 | 3.2513 | 13.1% | 否 |
| asr | 20.000s/clients=32 | latency_s_p95 | 3.2918 | 3.7567 | 14.1% | 否 |
| asr | 30.000s/clients=1 | throughput_x_median | 40.6662 | 35.2896 | 13.2% | 否 |
| asr | 30.000s/clients=1 | first_text_s_p95 | 0.1526 | 0.1623 | 6.3% | 否 |
| asr | 30.000s/clients=1 | latency_s_p95 | 0.7469 | 0.8585 | 15.0% | 否 |
| asr | 30.000s/clients=2 | throughput_x_median | 72.6750 | 63.2377 | 13.0% | 否 |
| asr | 30.000s/clients=2 | first_text_s_p95 | 0.2454 | 0.2587 | 5.4% | 否 |
| asr | 30.000s/clients=2 | latency_s_p95 | 0.8487 | 0.9621 | 13.4% | 否 |
| asr | 30.000s/clients=4 | throughput_x_median | 125.6600 | 110.2460 | 12.3% | 否 |
| asr | 30.000s/clients=4 | first_text_s_p95 | 0.3508 | 0.3711 | 5.8% | 否 |
| asr | 30.000s/clients=4 | latency_s_p95 | 0.9724 | 1.0938 | 12.5% | 否 |
| asr | 30.000s/clients=8 | throughput_x_median | 199.1181 | 178.0978 | 10.6% | 否 |
| asr | 30.000s/clients=8 | first_text_s_p95 | 0.5629 | 0.6002 | 6.6% | 否 |
| asr | 30.000s/clients=8 | latency_s_p95 | 1.2057 | 1.3467 | 11.7% | 否 |
| asr | 30.000s/clients=16 | throughput_x_median | 201.6741 | 181.8629 | 9.8% | 否 |
| asr | 30.000s/clients=16 | first_text_s_p95 | 1.7744 | 1.8755 | 5.7% | 否 |
| asr | 30.000s/clients=16 | latency_s_p95 | 2.4514 | 2.6307 | 7.3% | 否 |
| asr | 30.000s/clients=32 | throughput_x_median | 205.2101 | 184.1520 | 10.3% | 否 |
| asr | 30.000s/clients=32 | first_text_s_p95 | 4.0175 | 4.4450 | 10.6% | 否 |
| asr | 30.000s/clients=32 | latency_s_p95 | 4.6734 | 5.2014 | 11.3% | 否 |

## 服务恢复与共驻

最终先启动ASR，再启动LLM，两个NPU模型同时驻留。四路LLM答案4/4正确，四路ASR转写4/4与基线逐字一致；共驻HBM采样最大值40021MiB。该检查用于功能验证，不作为独占TPS。原始记录为 [cohost-smoke.json](/home/qwen-upgrade-20261015/cohost-smoke.json)。

LLM服务为qwen-llm.service，监听127.0.0.1:8100，默认原14B；ASR服务为qwen-asr-ascend.service，监听50011。8000的其他服务未改动。两项API就绪后才报告恢复。模型切换与新模型量化报告见 [四款模型报告](quantized-models.md)。

## 原始数据与回退

最终验收根目录 `/home/qwen-upgrade-20261015/final-verification/`：baseline符号链接指向原始旧环境数据；candidate保存最终14B和ASR记录，logs保存完整启动与长输入日志；comparison.json/md包含83项比较；asr-transcripts.json包含816条文本比较。原先candidate和调优日志也完整保留，不覆盖历史试验。

默认LLM入口 `bash scripts/run_ascend.sh`；回退入口 `QWEN_RUNTIME=legacy bash scripts/run_ascend.sh`。ASR回退入口 `QWEN_ASR_RUNTIME=legacy bash scripts/serve_ascend.sh`。切换前停止原实例。legacy入口已清理新CANN库路径，并显式选择原Poetry环境，避免XDG缓存路径影响环境选择。

新LLM环境：`/home/qwen-upgrade-20261015/virtualenvs/qwen-llm-INLN9vSh-py3.10`；新ASR环境同目录下qwen-asr-nnyoWybn-py3.10。旧LLM在 `/home/qwen-llm-build/virtualenvs/qwen-llm-INLN9vSh-py3.10`，旧ASR在 `/root/.cache/pypoetry/virtualenvs/qwen-asr-nnyoWybn-py3.10`。

# CANN 9.1.1 / Torch 2.9.1 升级验收

2026-10-15，Ascend910B4-1（物理4、逻辑0，64 GiB），驱动26.1.1。两个项目均保留新环境，原环境和CANN仍完整保留。本次采用比用户要求更细的验收：相同负载下吞吐下降或P95总延迟/首token延迟增加超过20%均视为不通过，超标组需再次复测；最终入选配置没有超标组。

| 组件 | LLM旧环境 | ASR旧环境 | 两项目新环境 |
| --- | --- | --- | --- |
| Python | 3.10.22 | 3.10.22 | 3.10.22 |
| Torch / torch-npu | 2.7.1 / 2.7.1 | 2.8.0 / 2.8.0 | 2.9.1 / 2.9.1 |
| vLLM | 0.11.0+empty | 0.13.0+empty | 0.18.0+empty |
| vLLM Ascend | 0.11.0rc1 | 0.13.0rc1 | 0.18.0+torch291.cann911 |
| CANN | 8.3.RC1 | 8.3.RC1 | 9.1.1 |
| Transformers | 4.57.1 | 4.57.6 | 4.57.6 |

Torch运行时标识为`2.9.1+cpu`，发行包metadata为`2.9.1`；实际计算使用torch-npu，CUDA为None。没有CPU模型回退。CANN在 `/home/qwen-upgrade-20261015/cann/` 并行安装，包括Toolkit、910B ops与NNAL；未覆盖旧CANN或更换驱动。进程加载的libascendcl、libopapi等路径已核对来自新CANN。

vLLM和Ascend均从固定源码构建，补丁和约束见 [deploy/ascend-291](../deploy/ascend-291/)。vLLM以`empty`设备构建，计算由Ascend插件接管。Torch 2.9.1组合经过本机测试，不等同于宣称上游原版安装矩阵支持。Triton-Ascend改用3.2.2以适应新CANN头文件，按照安装包依赖先安装基础Triton 3.5.0、最后覆盖[Ascend实现](https://gitcode.com/Ascend/triton-ascend)；两环境`pip check`通过。

LLM沿用原14B W8A8，输入8192、输出8192、4活动请求、2048批处理token、10.25 GiB KV。入选FULL图模式，捕获`[1,2,4,64,128,256]`，异步调度、npugraph_ex、前缀缓存开启，支持4.09路完整16384-token序列。短请求生成128 token，各并发三轮；8并发部分请求按约定返回429，属于过载验证，不纳入吞吐退化门槛。

新环境40/40固定答案题通过，4/4不同8192-token材料检索通过；自由文本不是逐字重现旧版本，3条历史冒烟中算术输出一致，另外两条措辞变化，原始结果已保留。能力题属于冒烟检查，不代表完整模型能力或量化精度评估。`</think>`的孤立前导控制标记由Responses适配层处理，未在评测器中放宽答案判定。

ASR共24组负载，每组三轮、816次请求全部成功，816条转写与旧环境逐字一致；最差吞吐下降15.10%，最差P95延迟增加16.19%，HBM采样峰值10509→9773 MiB。ASR注意力/RoPE的4项真实NPU数值对齐测试通过。详细结果见 [ASR报告](/root/repositories/qwen-asr/docs/cann911-torch291.md)。

调优过程的失败配置同样保留：PIECEWISE使LLM吞吐下降31%～33%；FULL_DECODE_ONLY使吞吐提升但短输入首token退化30%～52%；关闭异步调度或npugraph_ex未解决首token问题。最终增加短输入FULL图捕获后同时改善吞吐与首token延迟。未将这些退步配置设为默认。noex冷缓存补测曾因切换配置中断，属于不完整记录，不用于验收。

| 并发 | 旧TPS | 新TPS | 吞吐变化 |
| ---: | ---: | ---: | ---: |
| 1 | 46.39 | 53.19 | +14.65% |
| 2 | 87.12 | 103.90 | +19.25% |
| 4 | 170.63 | 200.68 | +17.61% |

## 逐项指标

吞吐下降超过20%或P95延迟增加超过20%的负载需额外复测；确认后回退。负值代表改善。

| 项目 | 负载 | 指标 | 旧环境 | 新环境 | 退化幅度 | 超过20% |
| --- | --- | --- | ---: | ---: | ---: | --- |
| llm | short/concurrency=1 | tokens/s | 46.3921 | 53.1865 | -14.6% | 否 |
| llm | short/concurrency=1 | seconds_p95 | 2.7646 | 2.4068 | -12.9% | 否 |
| llm | short/concurrency=1 | ttft_seconds_p95 | 0.0634 | 0.0416 | -34.4% | 否 |
| llm | short/concurrency=2 | tokens/s | 87.1246 | 103.8981 | -19.3% | 否 |
| llm | short/concurrency=2 | seconds_p95 | 2.9924 | 2.4733 | -17.3% | 否 |
| llm | short/concurrency=2 | ttft_seconds_p95 | 0.1416 | 0.0675 | -52.3% | 否 |
| llm | short/concurrency=4 | tokens/s | 170.6291 | 200.6789 | -17.6% | 否 |
| llm | short/concurrency=4 | seconds_p95 | 3.0449 | 2.5809 | -15.2% | 否 |
| llm | short/concurrency=4 | ttft_seconds_p95 | 0.1445 | 0.0736 | -49.0% | 否 |
| llm | 8192-input/hot/concurrency=4 | tokens/s | 123.3759 | 141.8057 | -14.9% | 否 |
| llm | 8192-input/hot/concurrency=4 | ttft_seconds_p95 | 0.2391 | 0.1940 | -18.9% | 否 |
| asr | 5.130s/clients=1 | throughput_x_median | 21.6816 | 18.9769 | 12.5% | 否 |
| asr | 5.130s/clients=1 | first_text_s_p95 | 0.1280 | 0.1423 | 11.2% | 否 |
| asr | 5.130s/clients=1 | latency_s_p95 | 0.2381 | 0.2746 | 15.3% | 否 |
| asr | 5.130s/clients=2 | throughput_x_median | 32.1161 | 28.9565 | 9.8% | 否 |
| asr | 5.130s/clients=2 | first_text_s_p95 | 0.2140 | 0.2331 | 9.0% | 否 |
| asr | 5.130s/clients=2 | latency_s_p95 | 0.3321 | 0.3690 | 11.1% | 否 |
| asr | 5.130s/clients=4 | throughput_x_median | 62.4647 | 55.6530 | 10.9% | 否 |
| asr | 5.130s/clients=4 | first_text_s_p95 | 0.2258 | 0.2452 | 8.6% | 否 |
| asr | 5.130s/clients=4 | latency_s_p95 | 0.3427 | 0.3832 | 11.8% | 否 |
| asr | 5.130s/clients=8 | throughput_x_median | 118.4117 | 105.4685 | 10.9% | 否 |
| asr | 5.130s/clients=8 | first_text_s_p95 | 0.2299 | 0.2503 | 8.9% | 否 |
| asr | 5.130s/clients=8 | latency_s_p95 | 0.3478 | 0.3913 | 12.5% | 否 |
| asr | 5.130s/clients=16 | throughput_x_median | 122.0721 | 109.0387 | 10.7% | 否 |
| asr | 5.130s/clients=16 | first_text_s_p95 | 0.5535 | 0.6114 | 10.5% | 否 |
| asr | 5.130s/clients=16 | latency_s_p95 | 0.6722 | 0.7526 | 12.0% | 否 |
| asr | 5.130s/clients=32 | throughput_x_median | 122.3627 | 111.1859 | 9.1% | 否 |
| asr | 5.130s/clients=32 | first_text_s_p95 | 1.2632 | 1.4009 | 10.9% | 否 |
| asr | 5.130s/clients=32 | latency_s_p95 | 1.3844 | 1.5433 | 11.5% | 否 |
| asr | 10.000s/clients=1 | throughput_x_median | 30.2865 | 26.5480 | 12.3% | 否 |
| asr | 10.000s/clients=1 | first_text_s_p95 | 0.1300 | 0.1404 | 8.0% | 否 |
| asr | 10.000s/clients=1 | latency_s_p95 | 0.3335 | 0.3781 | 13.3% | 否 |
| asr | 10.000s/clients=2 | throughput_x_median | 48.5905 | 43.0569 | 11.4% | 否 |
| asr | 10.000s/clients=2 | first_text_s_p95 | 0.2121 | 0.2290 | 7.9% | 否 |
| asr | 10.000s/clients=2 | latency_s_p95 | 0.4274 | 0.4772 | 11.7% | 否 |
| asr | 10.000s/clients=4 | throughput_x_median | 91.8795 | 81.7750 | 11.0% | 否 |
| asr | 10.000s/clients=4 | first_text_s_p95 | 0.2321 | 0.2488 | 7.2% | 否 |
| asr | 10.000s/clients=4 | latency_s_p95 | 0.4440 | 0.4970 | 11.9% | 否 |
| asr | 10.000s/clients=8 | throughput_x_median | 172.2324 | 154.1188 | 10.5% | 否 |
| asr | 10.000s/clients=8 | first_text_s_p95 | 0.2506 | 0.2626 | 4.8% | 否 |
| asr | 10.000s/clients=8 | latency_s_p95 | 0.4705 | 0.5195 | 10.4% | 否 |
| asr | 10.000s/clients=16 | throughput_x_median | 175.0279 | 159.8085 | 8.7% | 否 |
| asr | 10.000s/clients=16 | first_text_s_p95 | 0.6924 | 0.7468 | 7.9% | 否 |
| asr | 10.000s/clients=16 | latency_s_p95 | 0.9269 | 1.0017 | 8.1% | 否 |
| asr | 10.000s/clients=32 | throughput_x_median | 157.4249 | 139.4440 | 11.4% | 否 |
| asr | 10.000s/clients=32 | first_text_s_p95 | 1.8194 | 2.0446 | 12.4% | 否 |
| asr | 10.000s/clients=32 | latency_s_p95 | 2.0356 | 2.3003 | 13.0% | 否 |
| asr | 20.000s/clients=1 | throughput_x_median | 38.0186 | 32.2793 | 15.1% | 否 |
| asr | 20.000s/clients=1 | first_text_s_p95 | 0.1429 | 0.1534 | 7.4% | 否 |
| asr | 20.000s/clients=1 | latency_s_p95 | 0.5385 | 0.6242 | 15.9% | 否 |
| asr | 20.000s/clients=2 | throughput_x_median | 64.8985 | 56.0842 | 13.6% | 否 |
| asr | 20.000s/clients=2 | first_text_s_p95 | 0.2284 | 0.2466 | 8.0% | 否 |
| asr | 20.000s/clients=2 | latency_s_p95 | 0.6441 | 0.7374 | 14.5% | 否 |
| asr | 20.000s/clients=4 | throughput_x_median | 122.1803 | 106.3659 | 12.9% | 否 |
| asr | 20.000s/clients=4 | first_text_s_p95 | 0.2503 | 0.2683 | 7.2% | 否 |
| asr | 20.000s/clients=4 | latency_s_p95 | 0.6714 | 0.7529 | 12.1% | 否 |
| asr | 20.000s/clients=8 | throughput_x_median | 184.3067 | 161.6669 | 12.3% | 否 |
| asr | 20.000s/clients=8 | first_text_s_p95 | 0.4425 | 0.4823 | 9.0% | 否 |
| asr | 20.000s/clients=8 | latency_s_p95 | 0.8673 | 0.9886 | 14.0% | 否 |
| asr | 20.000s/clients=16 | throughput_x_median | 188.4595 | 166.6003 | 11.6% | 否 |
| asr | 20.000s/clients=16 | first_text_s_p95 | 1.2672 | 1.4108 | 11.3% | 否 |
| asr | 20.000s/clients=16 | latency_s_p95 | 1.7030 | 1.9243 | 13.0% | 否 |
| asr | 20.000s/clients=32 | throughput_x_median | 194.1206 | 169.5224 | 12.7% | 否 |
| asr | 20.000s/clients=32 | first_text_s_p95 | 2.8741 | 3.2851 | 14.3% | 否 |
| asr | 20.000s/clients=32 | latency_s_p95 | 3.2918 | 3.8059 | 15.6% | 否 |
| asr | 30.000s/clients=1 | throughput_x_median | 40.6662 | 34.8683 | 14.3% | 否 |
| asr | 30.000s/clients=1 | first_text_s_p95 | 0.1526 | 0.1630 | 6.8% | 否 |
| asr | 30.000s/clients=1 | latency_s_p95 | 0.7469 | 0.8678 | 16.2% | 否 |
| asr | 30.000s/clients=2 | throughput_x_median | 72.6750 | 61.7325 | 15.1% | 否 |
| asr | 30.000s/clients=2 | first_text_s_p95 | 0.2454 | 0.2660 | 8.4% | 否 |
| asr | 30.000s/clients=2 | latency_s_p95 | 0.8487 | 0.9853 | 16.1% | 否 |
| asr | 30.000s/clients=4 | throughput_x_median | 125.6600 | 110.1095 | 12.4% | 否 |
| asr | 30.000s/clients=4 | first_text_s_p95 | 0.3508 | 0.3738 | 6.6% | 否 |
| asr | 30.000s/clients=4 | latency_s_p95 | 0.9724 | 1.1051 | 13.7% | 否 |
| asr | 30.000s/clients=8 | throughput_x_median | 199.1181 | 177.3327 | 10.9% | 否 |
| asr | 30.000s/clients=8 | first_text_s_p95 | 0.5629 | 0.5976 | 6.2% | 否 |
| asr | 30.000s/clients=8 | latency_s_p95 | 1.2057 | 1.3522 | 12.1% | 否 |
| asr | 30.000s/clients=16 | throughput_x_median | 201.6741 | 175.8736 | 12.8% | 否 |
| asr | 30.000s/clients=16 | first_text_s_p95 | 1.7744 | 1.9576 | 10.3% | 否 |
| asr | 30.000s/clients=16 | latency_s_p95 | 2.4514 | 2.7354 | 11.6% | 否 |
| asr | 30.000s/clients=32 | throughput_x_median | 205.2101 | 178.4061 | 13.1% | 否 |
| asr | 30.000s/clients=32 | first_text_s_p95 | 4.0175 | 4.5988 | 14.5% | 否 |
| asr | 30.000s/clients=32 | latency_s_p95 | 4.6734 | 5.3791 | 15.1% | 否 |


原始报告根目录：`/home/qwen-upgrade-20261015/`。baseline/和candidate/保存相同负载原始记录；logs/保存长输入结果、各调优配置与启动日志；comparison.json是完整验收数据；acceptance.json记录保留新环境的决定。8K表格比较的是至少三轮热缓存结果；冷启动/冷缓存记录单独保留，不将冷缓存与热缓存混比。

启动：`bash scripts/run_ascend.sh` 默认选择新环境与入选14B图配置；`QWEN_RUNTIME=legacy bash scripts/run_ascend.sh`选择旧Torch 2.7.1 / CANN 8.3.RC1 / PIECEWISE。新环境根目录可通过`QWEN_RUNTIME_ROOT`设置。新CANN的安装脚本与源码快照在数据盘升级目录，原CANN位于`/usr/local/Ascend/ascend-toolkit/8.3.RC1`。

多模态架构只加载文本时，额外回补 [Ascend 上游 PR #14618](https://github.com/vllm-project/vllm-ascend/pull/14618)，修复预热与真实请求输入路径不一致。该补丁不改变原14B和ASR的模型配置。新模型验证及下载记录见 [量化模型报告](quantized-models.md)。

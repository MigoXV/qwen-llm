# NVIDIA 量化与实测

本目录复用原 Ascend HTTP/SSE 计时和质量题库，保留量化导出、TPS、质量和容量验证工具。每个实例测并发 1、2、3、4，请求并发数不表示卡数。实验参数不进入生产启动入口。

当前 GPU 2 保留 [24 GiB 两路服务](../../../docs/nvidia-gpu2-service.md)。NVIDIA 入口默认监听 `0.0.0.0`，可用 `--host` / `QWEN_HOST` 覆盖。下列命令中的设备编号是历史测试示例，执行前须按实际分配调整，避开运行中的服务。算子统一使用 vLLM 原生实现，未采用的手写候选和专用 profiling 工具已清理。

## 独立环境

Python 3.11 的推理与量化环境分别位于 `/workspace/qwen-nvidia/runtime` 和 `/workspace/qwen-nvidia/quant`，不覆盖 Ascend 的 Python 3.10 环境。首次安装：

```bash
QWEN_NVIDIA_ENV=runtime QWEN_PYTHON=/path/to/python3.11 bash scripts/install_nvidia.sh
QWEN_NVIDIA_ENV=quant QWEN_PYTHON=/path/to/python3.11 bash scripts/install_nvidia.sh
```

`deploy/nvidia/` 固定核心包版本，完整实际依赖保存在各环境的 `installed.txt`。运行命令前设置 `QWEN_INSTANCE` 再加载环境，避免实例共享编译缓存。

## 校准及导出

```bash
export QWEN_NVIDIA_ENV=quant
source scripts/env_nvidia.sh
poetry run python -m tests.performance.nvidia.calibration \
  --model /workspace/model-bin/windowsxp811203/Qwen3.8-27B-Abliterated \
  --output outputs/FRESH/calibration.jsonl
CUDA_VISIBLE_DEVICES=1 poetry run python -m tests.performance.nvidia.quantize \
  --source /workspace/model-bin/windowsxp811203/Qwen3.8-27B-Abliterated \
  --output /workspace/model-bin/windowsxp811203/FRESH-W4A16-NVIDIA \
  --calibration outputs/FRESH/calibration.jsonl --method awq-gptq
```

候选使用对称 INT4 group128 权重、BF16 激活。`awq-gptq` 先执行 AWQ 再用 GPTQ；`gptq` 为独立配方。视觉模块、输入词嵌入、输出头、GDN 的 a/b 投影及 MTP 不量化。AWQ 激活缓存卸载到 CPU，避免 40GB 卡校准时内存不足。校准输入固定为同一 revision 的前 512 条有效样本，最大 4096 token；两配方必须共用同一 JSONL 和哈希。

原权重只读。每次使用全新输出目录，未完成目录带 `.quantization.incomplete`；导出完成不等于质量验收通过。产物保存配方、软件版本和所有权重分片哈希，未由 HF 建模的 MTP 权重从源文件保留。

导出后执行 `poetry run python -m tests.performance.nvidia.audit_artifact /path/to/MODEL --output outputs/FRESH/artifact-audit.json`，从实际 safetensors 检查索引、INT4 打包、分片哈希、MTP 和分词文件。可用 `--snapshot /path/to/source-manifest.json` 对照原模型的事前哈希快照；产物完整性检查与推理质量验收分别报告。

## 双卡主实例

```bash
export QWEN_NVIDIA_ENV=runtime QWEN_INSTANCE=main
source scripts/env_nvidia.sh
CUDA_VISIBLE_DEVICES=2,3 NCCL_P2P_DISABLE=1 poetry run qwen-llm serve-nvidia \
  --model-path /workspace/model-bin/windowsxp811203/MODEL \
  --model-name qwen38-nvidia --port 8120 --max-model-len 262144 --cuda-graph
```

本机 A100 PCIe 的默认 P2P 初始化实测挂起，因此两卡对照统一使用 `NCCL_P2P_DISABLE=1`，并禁用 vLLM 自定义 all-reduce。这属于本机通信配置，不应直接推广到其他机器。BF16 原权重测试先用 `--max-model-len 32768`；量化版本能否支持更长上下文由实测决定。

## 单卡 FP8 KV 实验

```bash
export QWEN_NVIDIA_ENV=runtime QWEN_INSTANCE=single-fp8
source scripts/env_nvidia.sh
CUDA_VISIBLE_DEVICES=1 poetry run python -m tests.performance.nvidia.cli serve \
  --model-path /workspace/model-bin/windowsxp811203/MODEL \
  --model-name qwen38-single-fp8 --port 8121 --tp 1 --context 262144 \
  --kv-dtype fp8_e4m3 --cuda-graph
```

普通生产入口不开放 FP8 KV。实验按 vLLM 0.32 的实际实现读取 checkpoint 的 KV scales；缺失时为 1.0，不能称为已校准。实际各层 scale、KV 存储字节数、图模式和进程身份在 `/health/ready` 记录。GDN 状态不因注意力 KV 使用 FP8 而自动变成 FP8。[vLLM FP8 KV 文档](https://docs.vllm.ai/en/v0.32.0/features/quantization/quantized_kvcache/)

`--context` 是输入与输出的总预算，`--max-output-tokens` 默认 8192，须小于 `context`。当前 GPU 2 服务配置为 131072 总上下文、32768 输出上限，输入上限因此为 98304；API 按实例配置校验，不再在协议 schema 中固定 8K。

在 BF16 KV 与 FP8 KV 对照中，可以通过实验参数 `--attention-backend FLASHINFER` 固定相同 backend；实际选择的后端与缓存 tensor dtype 由 Worker 记录，避免将后端切换误认为单纯的 KV 精度收益。

默认显存预算为 0.9。实验入口支持 `--gpu-memory-utilization`（最高 0.98），用于另行验证提高预算后的容量；更高预算的结果必须单列配置和完整生成证据，不能直接把默认预算的边界称作显卡物理上限。

实验入口与生产入口均支持 `--kv-cache-gib`，显式指定每卡 KV 预算，优先于 vLLM 根据显存比例自动计算的池大小。首次编译与命中编译缓存的启动路径可能测出不同的临时内存峰值，导致相同显存比例分配不同的 KV 池；需要可复现容量时，应固定预算并在全新编译缓存下完成启动和完整生成验证。实际池按页取整，仍以 `/health/ready` 中的字节数和容量记录为准。

## TPS 与质量

```bash
poetry run python -m tests.performance.nvidia.cli benchmark \
  --base-url http://127.0.0.1:8120 --output outputs/FRESH/short.json
poetry run python -m tests.performance.nvidia.cli benchmark \
  --base-url http://127.0.0.1:8120 --tokens 512 --rounds 5 --output outputs/FRESH/generation.json
poetry run python -m tests.performance.nvidia.cli evaluate \
  --base-url http://127.0.0.1:8120 --dataset /path/to/gsm8k-test.jsonl \
  --math-samples 1319 --output outputs/FRESH/quality.json
poetry run python -m tests.performance.nvidia.long_context \
  --base-url http://127.0.0.1:8120 --output outputs/FRESH/8k-cold \
  --lengths 8192 --tokens 128 --rounds 3
```

总 TPS = 本轮所有成功请求实际输出 token 总数 / 本轮墙钟时间；单请求 decode TPS = `(实际输出 token - 1)/(末包时间 - 首个非空增量时间)`。TTFT 从发出 HTTP 请求计到首个非空增量。预热不计入统计，不把 SSE 包数当成 token 数。每轮保留原始响应、TTFT、输出长度、GPU 进程和其他设备负载。两实例同期运行时 CPU、内存、PCIe 仍可能互相影响。

使用 `poetry run python -m tests.performance.nvidia.dual --output outputs/FRESH/paired` 同步两实例的每轮开始时刻；两边均按 1、2、3、4 路测量。附加 `--seconds 3600 --rounds 1 --tokens 512` 可重复矩阵至少一小时，逐批保存结果。固定对端权重、KV 和负载，再比较主实例 BF16 与量化版，避免将共享资源争用混入量化收益。

双实例矩阵保存前后的 Prometheus 指标，结束后等待统计刷新再记录。这个等待不计入 TPS 和已完成的稳定性时长，用于核对抢占与请求状态；资源盘点同时保留 Worker、Engine Core、API 的 RSS/FD 及 GPU 温度、时钟、功耗。

定位 RSS 增长时，可在相同请求排空边界执行 `poetry run python -m tests.performance.nvidia.rss PID1 PID2 --output outputs/FRESH/memory.json`，保存 `/proc` 的完整映射及 RSS、PSS、私有／共享页。比较时将同路径映射合并，不直接按包含终止地址的映射区间作差，因为堆扩展会改变区间。不要仅凭 RSS 增长将分配器预热或共享页驻留认定为存活对象泄漏。

vLLM 0.32.0 的 SHM 忙循环通知积压及 NVIDIA 专用修复见 [RSS 定位报告](../../../docs/nvidia-rss.md) 和 [环境补丁](../../../deploy/nvidia/README.md)。测试记录需同时保留补丁清单，包版本号不能区分修复前后；诊断忙循环问题时还须检查采样或屏障是否产生了足以让通知队列自行清理的空闲间隔。

`poetry run python -m tests.performance.nvidia.behavior --output outputs/FRESH/behavior.json` 保存 20 道正常敏感提示的回答与启发式拒答标记，供原版/量化版人工核对；该标记不是安全合规率或综合能力分数。

质量沿用 40 道固定题、64 道项目代码题和完整 GSM8K 1319 题；代码题不代表 HumanEval。默认 greedy、非思考，GSM8K 输出上限 2048。比较要求相同题目、数据集及判题器哈希。原基线通过的固定题不新增失败，代码通过数不减少，GSM8K 下降不超过 1 个百分点，作为本项目采用门槛。

## 长上下文容量

容量服务改用实验入口并添加 `--force-length --kv-dtype auto --tp 2`（双卡），或 `--force-length --kv-dtype fp8_e4m3 --tp 1`（单卡）。强制输出仅存在于实验子类，不能拿该服务做自然停止的质量测试。

```bash
poetry run python -m tests.performance.nvidia.capacity \
  --base-url http://127.0.0.1:8121 --output outputs/FRESH/single-capacity
poetry run python -m tests.performance.nvidia.long_context \
  --base-url http://127.0.0.1:8120 --output outputs/FRESH/retrieval \
  --lengths 122880,253952 --tokens 32 --rounds 3 --retrieval
```

容量的上下文包含输入和输出，默认保留 8192 输出 token。先测 16K、32K、64K、128K、192K、256K，再在失败区间按 8K 分辨率定位。每档并发分别报告最大完整通过长度与最小失败长度。通过要求所有请求达到精确输入和完整输出、设备归属正确、实例不变且没有抢占；检测到抢占立即取消该轮，避免将重算/排队成功误写为显存同时容纳。

已经完成短请求验证时，可用 `--start-context 131072` 从目标 128K 开始向上探测；若首个点失败，仍会向下二分，定位同样的 8K 边界。每个通过点仍须实际完成全部 8192 输出 token。

复验固定预算时，可用 `--from-allocated-pool` 从真实池能同时驻留的最大候选长度开始，实际生成完整输出，再测试下一档。若候选点发生抢占则逐档下降；池大小只是选择测试点的依据，不替代实际生成证据。默认步长为 8192 token，达到模型原生上限时不越界测试。

还必须满足实际 Worker 分配池在对应长度的容量不小于请求并发数。该容量使用 vLLM 自身的缓存页算法从真实池大小计算，包含 GDN 状态和页对齐；即使服务通过排队完成全部请求，也不会被当作完整并发驻留。

检索是独立质量测试，覆盖开头、中间、结尾和不同校验码。热缓存测试需另启用 `--prefix-caching`，使用 `--cache hot --rounds 7`，比较最后三轮；不要把关闭缓存的重复请求称为热缓存命中。

执行检索前须重启为不带 `--force-length` 的服务，保证校验码可以自然停止；否则测试入口会拒绝执行。

## 24 GiB 与词表压缩

本轮结果见 [24 GiB 容量与质量报告](../../../docs/nvidia-24g-hotspots.md)。词表矩阵单独使用 group128 对称 RTN；原 AWQ/GPTQ 主干逐张量验证不变，不能把新产物称为经过 AWQ/GPTQ 校准的词表量化。

```bash
CUDA_VISIBLE_DEVICES=2 poetry run python -m tests.performance.nvidia.vocab_quantize \
  --source /path/to/W4A16-AWQ-GPTQ --output /path/to/FRESH-VOCAB4 --bits 4
```

新产物需要 `scripts/patch_nvidia.py` 的词嵌入接线补丁。导出与推理各占一张指定 GPU，必须先结束同卡的其他实验。INT8 也是导出选项，本轮实际推理测试只覆盖 INT4 词表。

# Qwen3.8 AscendC W4A16 V2

本轮在 Ascend 910B4-1 上实现并测试更细粒度的 FFN 算子优化。正式服务默认仍使用原生后端；实验只通过独立 Worker 显式安装。V1 的报告与原始记录保留在 [原报告](qwen38-ascendc.md)。

原始实验目录：`/home/qwen-models/reports/qwen3.8-27b/marlin-v2-20261103`。

## 本轮结果：先对齐，再短测速度

根据测试范围的修正，本轮最终采用短算子筛选。此前启动的完整质量测试已中止，部分记录保留但不算完整基线；没有继续完整 GSM8K、23 场景、两次全矩阵或候选整模型测试。

最终短测进程用时 **56.1 秒**，覆盖层 0 的两种投影、M=3/12/1536、两种布局，共 12 个形状/布局组合、72 个配置。**72/72 数值与图重放通过**；包含零输入、64 倍幅值及 M=12 FP32 oracle 的最大相对 L2 误差为 **0.002311**，原生回退全部通过。

每个场景的最佳候选仍需原生的 **1.52～8.42 倍耗时**，因此停止该方向的长测，正式默认保持原生。以下只代表本次短测范围，不能替代全部形状的验收。

| 投影 | M | 原生 NZ 布局候选/原生耗时 | 专用布局候选/原生耗时 |
|---|---:|---:|---:|
| gate_up_proj | 3 | 4.507 | 3.210 |
| gate_up_proj | 12 | 7.855 | 3.667 |
| gate_up_proj | 1536 | 1.972 | 1.521 |
| down_proj | 3 | 8.422 | 3.942 |
| down_proj | 12 | 5.015 | 2.332 |
| down_proj | 1536 | 1.965 | 1.518 |

比值为候选耗时/配对原生耗时，越小越快。单次输出应数值对齐，但 Split-K 改变浮点累加顺序，因此保留误差检查；这不要求先运行整模型质量集。不同布局行的原生时间各自配对，原始五轮记录保留在结果文件中。

机器可读结果见 [V2 短测汇总](qwen38-ascendc-v2-results.json)，完整原始数据见实验目录 `quick/result.json`。可通过 `summarize_ascendc_v2 实验目录 --quick` 重建汇总。下述完整矩阵和服务门槛是工具具备的后续流程，本轮没有执行到这些阶段。

## 实现范围

[Marlin](https://github.com/IST-DASLab/marlin) 的离线布局、异步搬运、反量化与矩阵计算重叠、分区归约是本轮的算法参考。Marlin 面向 CUDA；这里按 Atlas A2 的存储层级和同步机制实现 AscendC 路径，没有移植 CUDA 指令或沿用其加速倍数。

| 层次 | 本轮实现 | 对照方式 |
|---|---|---|
| group128 反量化 | 成批加载 INT4，复用 BF16 尺度；显式 INT4→FP16→FP32 乘尺度→BF16 | 反量化、预反量化 BF16 GEMM、完整两段式算子分别计时 |
| K 流水 | GM 环形 B 缓冲深度 2/3；AIV 生产、AIC 消费；整段 K 结束才输出 FP32 L0C 累加结果 | N tile 64/128，K tile 128/256/512，M tile 16/64/128/256 |
| 并行 Split-K | 一个混合内核内分区 K；FP32 部分结果；一个归约内核 | 小 M 测 S=1/2/4/8，大 M 固定 S=1 |
| 专用权重布局 | `blocked64x128-v1`：外层 `[N/64,K/128]`，内部按 BF16 nZ 的 `[K16,N16,Ninner,Kinner]` 顺序打包 INT4 | 与原生 NZ 同权重、同尺度、同输入独立测试 |
| 模型安装与回退 | 仅安装两次独立三层复核共同通过的同一配置；未命中形状走原生算子 | 专用布局先逆打包至共享 NZ scratch，转换计入回退耗时 |

复用安装环境中的 CATLASS TileCopy、TileMmad 与 BlockMmad，没有修改 CATLASS 源码。Atlas A2 的本实现通过 GM 交换 AIV 反量化后的 B tile；不能将它描述为 UB 直达 L1 或零 GM 中转。

专用布局替换实验 Worker 中被选中投影的权重存储，不保留全模型双份设备权重。原始 safetensors 文件保持原样。两类投影的共享 NZ scratch 合计 127.5 MiB。独立算子测试的固定私有工作区为 620,603,392 字节，V1 对照另用 170 MiB；这些是测试资源统计，不能代替候选服务的实际 HBM 增量测量。

## 验证工具与计时方法（本轮只执行短测）

- 实际层 0/30/63、gate_up/down 两种投影，M 为 1、3、6、12、38、117、1536、2048、6144、7680、8192；每种布局 66 个组合。
- 层 0 调优后每个投影/布局/M 保留两个配置；两次独立进程扩展到完整三层矩阵。
- 每个配置验证随机、零输入、64 倍幅值；M=1/12 另与 FP32 oracle 对齐；相对 L2 容差 0.01。
- 图捕获后以 NaN 覆盖输出，再重放验证，避免把空捕获计为正确。每次保留 5 组交替的原生/候选时间。
- 主计时每张图含 16 次调用（M≤117）或 4 次调用，事件时间除以调用数；另保存单次图重放时间。该时间包含设备图调度，不声称是纯内核耗时。
- V1 和原生使用同一新方法重新计时，不能直接拿旧报告的 0.625 比值判断 V2 收益。
- 原生 NZ 与专用布局均验证原生回退；专用布局的逆转换要求物理 NZ 字节逐位一致。

调试中发现 SDK 在 Atlas A2 上以 `sizeof(int4b_t)==1` 计算 `DataCopy` 的 count。直接传 INT4 元素数会多搬一倍，覆盖 UB 后续缓冲。V2 和用于新计时的 V1 均改为显式字节搬运。旧构建、失败日志和快照保留，没有覆盖旧测量。

最终构建的冒烟检查包含 116 个算子/布局配置，全部通过 eager 与图重放；两种投影、两种布局的首/中/末 K group 反量化值均与 CPU 逐位一致。

## 保留的完整验收工具（本轮未进入）

算子迭代的默认顺序为有限数值对齐、短微基准、有收益后的少量服务冒烟。以下是原计划保留的完整验收能力，只有另有完整验收需求时才显式调用，不作为每轮算子优化的前置任务。

算子安装要求同一个配置在两次独立进程、全部三个验证层均至少快 10%。筛选记录绑定模型配置、Python 源码、原始算子库、主机桥和依赖环境哈希。缺少完整覆盖、回退、幅值、FP32 或图重放证据时拒绝安装。

通过算子门槛才进入候选服务检查：两种布局先测六个核心场景各 5 轮及 40 个固定回答、64 个代码题、128 个 GSM8K；筛选优胜者再做 23 场景、完整 1319 个 GSM8K、72 个长输入、72 个块边界、68 个前缀增长和一小时稳定性检查。

服务须同时满足生成 TPS 几何平均比值≥1.05、冷首字延迟几何平均比值≤0.95，每个场景 TPS≥0.95、长输入首字延迟≤1.05。完整 GSM8K 最多下降 1 个百分点，固定题无新增失败，代码题通过数不下降，四路 64K 能力保持，实际额外 HBM≤1 GiB。实验后另做原生六场景漂移复测。即使通过也不自动更改正式默认配置。

## 复现入口

先使用项目 Poetry 环境和 `scripts/env_ascend_291.sh`，确保目标 NPU 没有其他模型进程。构建及结果目录必须不存在。

```bash
bash tests/performance/qwen38/kernels/build.sh /absolute/new-build w4a16_v2.cpp
poetry run python -m tests.performance.qwen38.build_ascendc_host \
  --source host_v2.cpp --library /absolute/new-build/libqwen_w4a16.so \
  --output /absolute/new-host
TASK_QUEUE_ENABLE=1 poetry run python -m tests.performance.qwen38.verify_ascendc_v2_smoke \
  --host-library /absolute/new-host/qwen_w4_host.so --output /absolute/new-smoke
TASK_QUEUE_ENABLE=1 poetry run python -m tests.performance.qwen38.verify_ascendc_v2 \
  --host-library /absolute/new-host/qwen_w4_host.so --phase quick --output /absolute/new-quick
# 短测有明确收益后，才使用 --phase tune 和两次完整 validate。
# 分别在两个新进程使用 --phase validate --screen /absolute/new-tune/result.json
poetry run python -m tests.performance.qwen38.select_ascendc_v2 \
  --first /absolute/validation1/result.json --second /absolute/validation2/result.json \
  --output /absolute/new-selection
```

`ascendc_v2_experiment` 控制器负责独占实验、不可变任务快照、设备执行超时、进度监督和最后一次服务恢复。队列文件给出精确 argv/env/timeout/progress；每个任务保留源码、revision、patch、stdout 和退出状态。恢复后必须检查真实 Responses 推理、单一正式 NPU 所有者及测试端口释放。

## 收尾核验

正式 8100 服务已恢复，真实 Responses 请求返回 `2`，NPU 上仅有一个正式 Worker，18100 测试端口已释放。ASR 服务保持停止。`poetry run pytest -q`：309 通过、13 跳过；`git diff --check` 通过。

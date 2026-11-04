# AscendC W4A16 实验

用于 Ascend910B/CANN9.1.1 的独立算子对照及显式服务实验，不自动替换生产后端。
保持现有 BF16、group128 和 INT4 权重语义，包含三类候选：

- 显式反量化后调用 CATLASS GEMM，可选 Split-K。各分区分别启动，再归约 FP32 输出。
- K 轴重排权重的混合 AIV/AIC 双缓冲流水。
- 直接读取原生 NZ 权重的混合 AIV/AIC 双缓冲流水，无持久化权重副本。

流水中 AIV 与 Cube 经两个有界 GM 槽交换数据，不宣称零搬运。磁盘权重按 N 轴打包，
前两类通过 `pack_k_int4` 无损重排；第三类保留原生 NZ 存储。合并尺度遵循生产加载的
BF16→FP32→BF16 舍入顺序。每次调用重新反量化；工作区及 Split-K 临时输出上限 1GiB。

```bash
bash tests/performance/qwen38/kernels/build.sh /home/NEW_BUILD
source scripts/env_ascend_291.sh
TASK_QUEUE_ENABLE=0 poetry run python -m tests.performance.qwen38.verify_ascendc_w4a16 \
  --library /home/NEW_BUILD/libqwen_w4a16.so --output /home/NEW_RESULT
```

输出目录必须不存在，设备必须无拥有进程。构建使用本机独立环境的 CATLASS 头文件，
可用 `QWEN_CATLASS_ROOT` 指向同版本源码。原始动态库显式加载，在当前 NPU stream
提交；独立验证关闭异步任务队列，并同步事件起点，避免 runtime 与框架队列交错。
不应将此 ctypes 调用方式直接放进正式模型。

验证使用 0/30/63 层的 gate_up/down 真实权重，以及 11 个 M 值，共 66 个形状。
每层均搜索 N tile 64/128、K tile 256/512；分阶段路径另搜索 Split-K 1/2/4/8。
超出工作区预算的配置明确跳过。所有候选验证随机、零和大幅值输入，relative-L2
上限 0.01，小 M 另与 FP32 参考比对。单算子设置 60 秒 runtime 执行超时。

计时前分别捕获原算子和候选 ACL Graph，把输出填成 NaN 后重放，拒绝空捕获。
预热后保留五次 NPU 事件跨度；反量化、归约及图下发间隙均计入，不能称为纯 kernel
时间。同一形状和配置必须在三层均至少降低 10% 耗时，才可进入后续模型筛选。
失败配置、精度误差、跳过原因、源文件及动态库哈希均保留；负结果也是有效实验结果。

服务实验使用 `host.cpp` 经 TorchNPU `OpCommand::RunOpApi` 提交，保留框架队列顺序：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 MAX_JOBS=2 poetry run python \
  -m tests.performance.qwen38.build_ascendc_host \
  --library /home/NEW_BUILD/libqwen_w4a16.so --output /home/NEW_HOST_BUILD
poetry run python -m tests.performance.qwen38.cli profile \
  --catalog configs/models.json --model-name qwen3.8-27b-optimized-v1 \
  --kv-cache-gib 22.5 --w4a8-linear-backend ascendc-w4a16 \
  --w4-selection /home/NEW_RESULT/result.json \
  --w4-host-library /home/NEW_HOST_BUILD/qwen_w4_host.so \
  --output-dir /home/NEW_SERVICE_RESULT
```

接入前核对数值、三层速度、图重放、模型配置、环境版本和库哈希，只接受合格的直接
NZ 配置；未命中形状继续调用现有原生算子。共享工作区约 170MiB，不缓存 BF16 权重。
主机调用计数只表示 eager/捕获调用，不能用来证明图重放次数，实际执行需结合设备 trace。
单算子通过后还必须完成服务质量、吞吐、冷首字延迟、容量、稳定性及基线漂移复测，
才能判断服务收益。实验入口不会修改正式服务别名或生产默认配置。

主机桥需要在异步队列开启时单独验证，设备必须空闲：

```bash
TASK_QUEUE_ENABLE=1 timeout --kill-after=5s 180s poetry run python \
  -m tests.performance.qwen38.verify_ascendc_host \
  --host-library /home/NEW_HOST_BUILD/qwen_w4_host.so --output /home/NEW_HOST_CHECK
```

这项检查覆盖两类投影、M=1/3/2048、原生回退及图重放，不进行速度筛选，也不产生
可用于模型接入的筛选记录。桥接回调不能调用会清空任务队列的 `stream()`；必须在
入队前用 `stream(false)` 获取原始句柄，否则队列会等待自身。入口设置60秒设备
执行超时，外层180秒超时同时约束主机等待；失败结果保留后使用新目录重测。

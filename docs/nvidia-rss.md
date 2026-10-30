# NVIDIA 单卡 RSS 增长定位

2026-10-29，使用物理 3 号卡定位，2 号卡作独立对照，每张卡一个实例。此前持续负载中的 Worker / Engine Core RSS 增长，已定位到 vLLM 0.32.0 的共享内存通信通知：忙循环读取共享内存时，ZeroMQ 通知队列没有及时触发合并清理。BF16 KV 也能复现，因此这项增长不能作为 FP8 KV 精度或显存泄漏的证据。

本文是 [NVIDIA 量化报告](nvidia-w4a16.md) 的后续调查；历史结果仍对应修复前环境，不改写其数值。结构化结果及原始文件 SHA256 见 [定位结果](nvidia-rss-results.json)。

## 原因与证据

`vllm/distributed/device_communicators/shm_broadcast.py` 的 `SpinCondition` 用 SHM 环形缓冲区传递实际数据，用 PUB/SUB socket 通知读者。原始 `record_read()` 只更新时间；只要持续读到数据，读者就忙循环访问 SHM，直到超过 1 秒没有读到数据才轮询通知 socket。

SUB 虽然配置了 `CONFLATE=1`，但读者长期不访问 socket 时，内部通知仍会积累。查询 `ZMQ_EVENTS` 才会触发合并清理，这是 [ZeroMQ 官方文档明确说明的行为](https://libzmq.readthedocs.io/en/latest/zmq_setsockopt.html#zmq-conflate-keep-only-last-message)。PUB 的 `SNDHWM=1` 也不能替代读端清理。这里增长的是 CPU 原生堆，不是 GPU KV cache，也不是 `/dev/shm` 映射持续扩容。

定位证据相互补充：

- 原始 FP8 KV、BF16 KV 新实例先跑短请求，后段 RSS 均可暂时稳定；先执行 253952-token 长输入，再连续跑 512-token 输出时出现增长，说明只跑一次短基准会漏掉问题。
- BF16 KV 的 16 批复现中，从首批到末批 Worker 增长 1584 KiB、Core 增长 744 KiB。它排除了“只有 FP8 KV 才有此问题”的解释，但两种精度的分配器历史不同，不能拿各自 RSS 绝对值直接比较。
- FP8 诊断实例的原生匿名映射 RSS：Worker 从 140 KiB 增至 456 KiB，Core 从 144 KiB 增至 460 KiB，各增长 316 KiB。Python `tracemalloc` 中 Core 存活分配基本稳定，不能解释同时间的原生页增长。
- 独立 CPU 最小复现只保留相同的 PUB/SUB、HWM、CONFLATE 配置，无模型、CUDA 或 KV cache。发送 60000 条通知且不访问读端，原生 malloc 在用内存从 1873568 增至 2391472 bytes（+505.77 KiB）。最后仅查询一次 `ZMQ_EVENTS`，在用量降到 1897552 bytes，释放 482.34 KiB；RSS 因分配器保留页面未立即下降。
- 相同 CPU 复现改为每约 256 条查询一次 EVENTS，原生在用内存只增加 22.77 KiB，约 10000 条以后保持稳定；RSS 从 21960 增至 22064 KiB 后不再随消息数量增长。版本为 libzmq 4.3.5 / pyzmq 27.2.0。

Python free list 和 malloc 高水位还会带来额外噪声：诊断实例 full GC 清除了约 832 KiB 的临时 Python 分配，`malloc_trim` 可使 Worker / Core RSS 分别下降约 46 / 51 MiB。这些是缓存和空闲页保留，不能替代通知队列的修复，也没有加入正式服务的每请求路径。

进一步直接使用 vLLM 的真实 `MessageQueue` 做修复前后 CPU 对照：先预热 1024 条，再持续传输 60000 条消息，两组均验证全部 **61024 条内容与顺序正确**。

| 实际 MessageQueue 版本 | 原生 malloc 在用量增量 | RSS 增量 | 最后手动查询 EVENTS 后的在用量 |
| --- | ---: | ---: | ---: |
| 原始 vLLM 0.32.0 | +545664 bytes（532.88 KiB） | +548 KiB | 159102480 bytes，降回预热水平 |
| 修复版 | −672 bytes | +16 KiB | 159100416 bytes，与结束时相同 |

这组测试加载了与 GPU 实验相同的两份 `shm_broadcast.py`，每万条记录一次内存；修复版从第 10000 条到第 60000 条原生在用量保持 159100416 bytes。它既复现增长，也通过单一代码变更消除增长，不依赖对 RSS 曲线的猜测。

## 修复范围

[补丁](../deploy/nvidia/patches/shm-notify-conflate.patch) 在读者线程每成功读取 256 条 SHM 消息后查询一次 `local_notify_socket.getsockopt(zmq.EVENTS)`。查询不消费取消消息；SHM 写入标记、读写顺序、空闲唤醒协议不变。没有增加后台线程、强制 GC、清空 KV 或重载模型。

[安装脚本](../scripts/install_nvidia.sh) 只在独立 NVIDIA runtime 环境应用补丁，Ascend 环境和量化工具环境保持原样。[补丁应用器](../scripts/patch_nvidia.py) 固定 vLLM 0.32.0，并检查原始文件和修复后文件 SHA256；重复运行幂等，遇到未知源码拒绝覆盖。详细使用方法见 [NVIDIA 环境说明](../deploy/nvidia/README.md)。

真实 CPU IPC 回归测试已通过 **3 项**：1025 条消息跨越多次清理边界，内容和顺序完全一致，并验证空闲后唤醒；阻塞读取可正常取消；连续消息流的原生堆增量小于 192 KiB。最后一项在冻结的原始版本上按预期失败（增长 540960 bytes），修复版全部通过，证明测试可以检出此次问题。测试通过 `CUDA_VISIBLE_DEVICES='' poetry run pytest tests/performance/nvidia/test_notify.py -q` 执行，不占 GPU。

## GPU 吞吐与带采样间隔的对照

物理 GPU 2 使用冻结的原始源码，GPU 3 使用修复版。两者均为 A100 PCIe 40GB、同一份 AWQ+GPTQ W4A16 权重、TP1、FP8 E4M3 KV、18 GiB KV 预算、262144 上下文上限、FLASHINFER、FULL_DECODE_ONLY CUDA Graph、batch tokens 2048、关闭前缀缓存和 MTP。两边均不启用诊断线程或 tracemalloc。

先执行同一份精确 253952-token 材料的中部校验码检索，均正确输出 `QZE7ADB279D7`，实际输出各 10 token。随后同步执行 16 批 1–4 路矩阵，每实例 160 个正式请求及 16 个预热请求均输出满 512 token，实例身份不变，设备始终独占。正式矩阵采用原项目相同 HTTP/SSE 计时方法，TPS 为每轮实际输出 token 总数除以本轮耗时，再取 16 轮中位数。

| 并发 | 原版总 TPS，GPU 2 | 修复版总 TPS，GPU 3 | 原版 TTFT ms | 修复版 TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 62.129 | 62.808 | 91.96 | 90.66 |
| 2 | 119.717 | 120.985 | 139.69 | 138.78 |
| 3 | 175.420 | 177.337 | 185.05 | 182.00 |
| 4 | 230.416 | 232.579 | 188.18 | 182.77 |

修复版相对原版约 +0.94%～+1.09%，未观察到吞吐回退。原版 GPU SM 时钟为 1395 MHz，修复版为 1395～1410 MHz；温度分别为 56～69°C、55～66°C，不能把这约 1% 的差异归因于补丁提速。开放式成语生成存在措辞差异，不把这组测试当作逐字输出一致性或完整质量基准；此次新增质量检查仅为上述同材料检索，完整模型能力结果仍见原报告。

| 从第 1 批到第 16 批的 RSS 变化 | 原版 | 修复版 |
| --- | ---: | ---: |
| Worker | +60 KiB | +64 KiB |
| Engine Core | +20 KiB | +24 KiB |
| API | +460 KiB | +248 KiB |

**这组带同步及完整 smaps 采样的结果不能单独证明 RSS 修复。** 两边都未持续增长。请求时间戳显示批次间到下一次预热存在约 1～2 秒的空闲，部分间隔即使考虑 1 秒时间戳精度仍超过 vLLM 的 1 秒忙循环阈值，足以触发原有通知清理。它适合验证 TPS，但会掩盖持续忙循环的队列问题。因此额外执行下节的连续负载，并直接测原生在用内存。

测量完成后已独立核验所有所属进程及端口释放；控制脚本曾把尚未 `wait()` 回收的 API zombie 误判为存活，处理记录保存在 `cleanup-note.json`，后续控制器已修正回收顺序。该问题发生在 16 批测量全部完成以后。

## 连续 GPU 负载验证

原始记录为 `outputs/nvidia-rss-continuous-20261029/`。使用相同 GPU 分配和模型配置，新启动两个实例，各连续提交 12 轮 4 路请求，每请求输出 2048 token。请求循环中不插入设备盘点、完整 smaps 采样或双实例屏障；两个实例都用相同的诊断线程每 5 秒读取 `mallinfo2` 和 RSS，该线程不访问通知 socket，不执行 GC / trim。结果只用于定位，不将诊断线程加入正式部署。

两边各 **48 个请求全部完成、输出 98304 token**。原版运行 424.365 秒，修复版 418.895 秒；客户端相邻轮次提交间隔最大仅 0.157 / 0.179 ms。两边实例身份不变、卡上没有其他模型进程。诊断线程同时记录实际加载的 `record_read` 文件路径，确认原版与补丁分别生效。

剔除前 60 秒的预热和结束前 10 秒之后的空闲清理区间，比较负载持续期间的采样。原版各进程窗口约 350～351 秒，修复版约 340～346 秒，分别有 71 和 69～70 个采样点；精确起止时间保存在 JSON。

| 进程 | 原版原生在用堆增量 KiB | 修复版原生在用堆增量 KiB | 原版 RSS 增量 KiB | 修复版 RSS 增量 KiB |
| --- | ---: | ---: | ---: | ---: |
| Worker | 1054.41 | 14.86 | 1340 | 172 |
| Engine Core | 899.38 | 32.33 | 868 | 32 |
| API | 151.33 | 105.72 | 172 | 60 |

按窗口全部采样点作线性拟合，Worker 原生堆增长速率由 **193.26 降到 1.48 KiB/min**，Core 由 **158.24 降到 6.75 KiB/min**。CPU 单消息队列对照与 GPU 连续负载一致支持：补丁消除了本次识别出的主要通知积压。API 和分配器仍有小幅变化；修复版 Worker 的 RSS 增量高于原生在用量，也说明 RSS 包含空闲页保留，不能只用 RSS 认定仍有同类存活对象增长。

本组使用连续数字提示及 2048 输出，和前节 512 输出的成语提示不同，吞吐不混合汇总。整个连续负载的总 TPS 为原版 **231.650**、修复版 **234.675**，每轮 TPS 中位数为 **231.941 / 234.720**；正式 1–4 路指标以前节为准。结束后两个实例全部退出，8132、8133 端口释放。

本次验证覆盖约 7 分钟连续负载与真实消息队列回归，足以复现并隔离已识别的问题；没有把修复前的一小时数据改称为修复后的一小时结果。

## 原始记录

- 首轮 FP8 / BF16 短负载：`outputs/nvidia-rss-20261029/`、`outputs/nvidia-rss-bf16-20261029/`。
- BF16 长输入后复现：`outputs/nvidia-rss-bf16-history-20261029/`。
- FP8 原生映射、Python 分配、GC / trim 诊断：`outputs/nvidia-rss-trace-20261029/`。
- CPU 最小复现：`outputs/nvidia-rss-20261029/zmq_repro.py`、`zmq-idle.json`、`zmq-events.json`。
- 实际 vLLM MessageQueue 的 CPU 对照：同目录 `shm_repro.py`、`shm-original.json`、`shm-fixed.json`。
- 最终修复前后同配置对照：`outputs/nvidia-rss-ab-20261029/`；GPU 2 使用冻结的原始通信源码，GPU 3 使用补丁，不启用 tracemalloc 或诊断线程。
- 连续 GPU 因果对照及原生采样：`outputs/nvidia-rss-continuous-20261029/`。
- 自动回归：`outputs/nvidia-rss-20261029/notify-tests-final.log`；原版负对照的预期失败记录为 `notify-test-original-expected-failure.log`。

FP8 KV 继续按用户要求作为实验选项；这次修复处理 CPU 通信内存增长，不改变此前容量或质量测试的适用范围，也不等于证明任意负载下长期无内存泄漏。

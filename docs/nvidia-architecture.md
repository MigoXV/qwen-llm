# NVIDIA 适配的架构决策

## 审查结论

本次复用已有 Engine 协议与 HTTP 层，收敛平台职责。输入模型计算、采样语义、Ascend 量化算法和专项优化保持不变。允许更换 NVIDIA 量化配方；这不意味着改变原始模型或 Ascend 路径。

| 优先级 | 原位置 | 问题与证据 | 决策 |
| --- | --- | --- | --- |
| P1 | engine/vllm.py 的 start | 通用命名的引擎启动同时拥有 Ascend 版本诊断、专属 Worker、MTP 与图配置 | 保留现有 V1Engine 的 Ascend 启动，将请求/生成/取消/关闭原样移至 AsyncGenerationEngine；CUDA 为同级实现 |
| P2 | config.py 的 ServingConfig | HTTP 容量和 NPU 专属选项共用一组字段，新平台容易继承不适用的默认值 | InferenceConfig 拥有共有配置；ServingConfig 保留旧 Ascend 契约，NvidiaConfig 拥有 TP/CUDA/KV 参数 |
| P2 | tests/performance/qwen38/benchmark.py | HTTP 计时与单 NPU 进程独占假设绑定 | 共享计时函数接受设备盘点函数；原入口保留旧默认，CUDA 只检查分配给目标实例的设备 |
| P1 | pyproject.toml 与运行时依赖 | vLLM 0.32 需要更新的 Python/FastAPI，不能覆盖原 Ascend Python 3.10 环境 | CUDA 解释器与工具虚拟环境分离；以 Python 版本标记保持旧 FastAPI 范围；平台依赖分别约束 |

## 权威来源与依赖

`CLI / typed config -> HTTP service -> Engine protocol -> platform engine -> vLLM worker -> model`

- HTTP 层拥有请求限制、协议映射、超时、取消与并发槽。
- 共享生成层拥有聊天模板、采样参数和生成事件映射，不导入设备插件。
- 平台引擎拥有依赖诊断、设备选择、vLLM 参数与实例观测。
- 已有 Ascend Worker、量化和专项适配继续拥有原有算法；CUDA Worker 只补充只读观测。
- 测试区拥有量化导出、FP8 KV 开关、强制生成长度、负载与验收。生产包不能导入测试区。
- 实验输出及模型清单记录实际版本、参数、哈希与结果，文档不取代实际观测。

## 不作为缺陷的现状

原 Ascend 实验通过测试子类注入 profiler，是已有的生产/实验隔离边界，不迁入生产引擎。已有模型专项补丁有明确硬件与版本约束，不抽象成跨平台算法。原协议与引擎之间的 Engine Protocol 已有明确职责，不引入额外调度器或通用插件注册框架。

## 迁移和回退

先核对原 Ascend start 的 AST、默认配置、服务协议，再提取共享方法；原命令和脚本继续可用。NVIDIA 独立环境与产物可单独撤回，不修改原权重。旧平台不能在本机执行真实 NPU 验收；本机通过的 CPU/协议回归与 NVIDIA 硬件证据分别报告。

## 初始双实例测试边界（历史）

物理 GPU 2、3 为双卡主实例，物理 GPU 1 为单卡实验。并发请求数与卡数分开记录。允许两实例同步运行，设备隔离只表示 GPU 所有权正确，不表示 CPU、内存及 PCIe 没有共享争用。FP8 KV 的速度和质量结论必须使用相同对端负载的 BF16 KV 对照。

## Responses 流式与前缀缓存（2026-10-29）

当前 GPU 2 为单卡常驻服务，GPU 3 为独立验证设备。工具请求曾把正文和参数缓冲到生成结束，导致 DSH 以首增量为起点的 TPS 严重虚高；引擎累计 token IDs 的计数没有重复累加。

`servicer/tool_stream.py` 仅适配 NVIDIA 环境已有的 vLLM Qwen parser，拥有每请求的增量解析状态，不加载模型、不执行工具。Responses 生成统一的文本、推理、函数输出项，Chat Completions 映射这些输出项，避免重新解析已转换的参数。`servicer/chat.py` 保留完整标记、工具选择与 schema 校验的权威规则；不完整或无效调用不能确认完成。Ascend 继续使用原有解析路径，不要求其环境安装新版 parser。

缓存由 vLLM 管理，部署配置只传递开关。NVIDIA Worker 报告实际开关、GDN 模式、分配池和驻留容量；`usage.input_tokens_details.cached_tokens` 取自引擎返回值。`prompt_cache_key` 只是兼容提示，实际复用按 token 前缀匹配，不增加会话存储或回答缓存。首字延迟、生成速度和输入缓存命中分别报告。

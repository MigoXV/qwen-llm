# Qwen3.8-27B 独立实验区

此目录包含压测、质量评测、量化导出、算子校验、profiling、结果归因和未验收的优化原型。生产包不导入此目录，主程序没有实验命令或 profiling 配置。普通 pytest 不加载模型；只有显式执行下面的硬件实验命令才会占用 NPU。

本轮仅验证 Ascend910B4 单卡、TP1、64层 dense Qwen3.8-27B，BF16 残差、FFN INT4 group128 / NZ 与现有其他混合精度。原始和当前量化模型保持只读。重新量化候选在 `/home/model-bin/Qwen/` 下建立独立目录，默认不发布到远端、不切换生产别名。

## 环境与入口

从仓库根目录执行：

```bash
source scripts/env_ascend_291.sh
poetry run python -m tests.performance.qwen38.cli --help
poetry run pytest -q
```

测试工具在进程内包装 vLLM 参数构造，设置独立 Worker 和 profiler；这些注入只在测试启动期间存在。模型子进程通过 `tests` 包导入实验模块，源码快照及 SHA256 写入每个实验目录。客户端独立进程驱动真实 Responses HTTP/SSE 服务。

测试专用环境变量示例位于本目录的 `.env.example`，主项目环境示例不再包含压测或GMM实验选项。CLI不自动读取这些文件，实验命令应显式给出参数。

## 热点复现与专项适配

```bash
poetry run python -m tests.performance.qwen38.cli profile \
  --output-dir /home/qwen-models/reports/qwen3.8-27b/FRESH_RUN \
  --kv-cache-gib 22.5 --suite comparison --rounds 5 --no-capture

poetry run python -m tests.performance.qwen38.cli profile \
  --output-dir /home/qwen-models/reports/qwen3.8-27b/FRESH_CANDIDATE \
  --kv-cache-gib 22.5 --optimizations scale,head,checkpoint \
  --suite full --rounds 5 --quality-samples 128 --no-capture

poetry run python -m tests.performance.qwen38.cli profile-report \
  /home/qwen-models/reports/qwen3.8-27b/FRESH_RUN
```

输出目录必须不存在。模型启动前检查卡上无拥有进程；开始测量后检查仅有目标 Worker；结束检查进程和端口释放。12.5 GiB 只支持约2.24路64K完整上下文，四路对照固定22.5 GiB。基线和候选的提示、实际 token 数、MTP、图模式及 KV 预算必须一致。

`scale` 在上游合并尺度之后一次舍入到BF16，消除每次 forward 的转换；`head` 校验内容后共享 target/MTP 输出头；`checkpoint` 进一步优先回收较早的GDN状态，保护MTP裁剪边界内各前缀最后可复用的检查点；已经通过的核心实现复用生产运行时。`cache`是未通过长文重放门槛的初版消融，在 Engine Core 使用未缓存空闲块优先的 O(1) 分配索引，缓存块仍遵守原LRU顺序，保留在测试区。候选只修改独立进程中的对象，不修改 site-packages 或全局安装。

`--no-capture` 用于计时；另行用 `--suite trace --cases ...` 抓取短 trace。`--quality-samples 1319` 执行全部本地 GSM8K，`--long-validation` 执行72个长文检索请求，`--stability-seconds 3600` 执行至少一小时混合负载。代码回归题为本项目固定64题及断言，不代表 HumanEval；生成代码在CPU时间和内存受限的独立子进程内校验。

完整候选还应使用`--cache-boundary-validation --prefix-growth-validation`，覆盖72个块边界检索和68个四路连续增长请求。前缀增长及稳定性阶段逐轮读取实际分配池，检查引用、空闲索引、检查点归属和元数据上界。`--quality-reference BASELINE/quality.json`在质量未达门槛时停止后续测量；比较要求相同数据集、提示词和代码判题器。

`compare.py`从原始交付时间计算混合负载三个短请求的批次间隔；一个交付批次可能包含多个token。`acceptance.py`核对完整质量、长文、五轮性能、至少3600秒稳定性及资源释放。`production_verify.py`通过普通生产CLI启动并测试HTTP、结构化输出、断连回收及五轮生成/57K热缓存，实际Worker必须来自生产包。

`verify_trace.py`要求目标FFN执行覆盖完整、全部Cast都有shape记录，才接受两类尺度Cast从128次/forward降至0。旧热点实例的端口释放以当时保存的`lifecycle-final.json`补充证明，不改写旧manifest。`verify_followups.py`分别检查六场景独立复测和三场景普通生产CLI对照；生产验收还要求四个请求实际各达到57344输入+8192输出。

```bash
poetry run python -m tests.performance.qwen38.verify_trace BASELINE_TRACE FINAL_TRACE \
  --output FRESH_ROOT/trace-acceptance.json
poetry run python -m tests.performance.qwen38.verify_followups FRESH_ROOT
poetry run python -m tests.performance.qwen38.verify_followups FRESH_ROOT --production
```

`finalize_model.py`只在完整验收、独立复测、trace、原始模型完整性及普通CLI基线就绪后写选中**新仓库**的元数据，拒绝原始目录和覆盖已有验收标记。`build_report.py`从通过的原始验收文件生成详细报告和机器结果，缺少证据即失败。

## 独立量化仓库

```bash
poetry run python -m tests.performance.qwen38.export_candidates
```

导出三个独立目录：`GDNW8-v1`、`HeadW8-v1`、`GDNHeadW8-v1`。前缀均为 `Qwen3.8-27B-W4A8-Ascend-`。只有指定浮点张量从原始BF16量化到INT8；其余张量保留当前量化版本。输出目录存在即拒绝覆盖，每个分片记录SHA256、来源和是否发生变化，未完成目录带 `.quantization.incomplete` 标记。导出完成不代表通过推理验收。

## 数值与采用门槛

```bash
poetry run python -m tests.performance.qwen38.verify_cached_scale \
  --output /home/qwen-models/reports/qwen3.8-27b/FRESH_NUMERICS.json
```

尺度测试使用真实0/30/63层融合gate/up及down权重，覆盖M=1/3/6/12/38/117/1536/2048/6144/7680，要求NZ格式、送入算子的BF16尺度及输出逐位一致。输出头以相同隐藏态比较 target/draft logits，同时保存实际存储指针。

质量门槛：基线通过的固定答案题不新增失败；GSM8K沿用推导并输出FINAL的提示，输出预算2048，避免将截断误算为量化错误。完整GSM8K准确率下降不超过1个百分点；代码合格数不低于基线；长文检索和并发隔离通过。性能至少五轮，独立新进程复测；主要吞吐提升5%，或长文热缓存TTFT下降20%，或混合卡顿下降50%，其他核心场景退化不超过5%。小收益优化单独报告重复性。

通过门槛后，必要执行适配位于`src/qwen_llm/engine/qwen38*.py`；测试入口复用这些实现，采集、导出和验证代码始终保留在此目录。失败候选保留记录且不进入生产默认配置。

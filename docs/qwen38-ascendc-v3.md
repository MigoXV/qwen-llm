# Qwen3.8 算子迭代与 native-tuned 短测

本轮在独立算子中确认了有收益的计算分支：对 M=6/12 的 FFN 投影启用 `npu_weight_quant_batchmatmul(inner_precise=1)`。原始 NZ 权重、BF16 尺度缓存与模型文件复用，不生成额外权重副本。

**服务结论已更正**：主程序接入时检查发现，旧版本在 Python 中按 M 分支，vLLM 的动态编译（`evaluate_guards=False`）将它固定到了默认路径。生成图有 128 个默认 W4 调用、0 个 `inner_precise=1` 调用。因此下表旧服务数据只保留作历史记录，不能证明加速收益；独立算子结果仍然有效。修复后的不透明算子分派及重新对照见 [主程序接入报告](qwen38-main-fast-w4.md)。原始证据保存在实验目录的 `folded-branch-evidence`。

同时实现并实测两个新的自定义 AscendC 内核版本；它们改善了原型的部分耗时，但尚未超过原生。

原始记录：`/home/qwen-models/reports/qwen3.8-27b/marlin-v3-20261103`。机器可读结果见 [短测汇总](qwen38-ascendc-v3-results.json)。

## 历史服务吞吐（加速分派未生效，不用于验收）

相同独占 910B4-1、模型与输入、MTP=2、KV=22.5 GiB、FULL_DECODE_ONLY、每请求 512 输出 token。每个场景预热一次，测量三轮，以下为中位数；吞吐按所有并发请求的 makespan 计算。

| 并发 | 原生 token/s | native-tuned token/s | 比值 |
|---:|---:|---:|---:|
| 1 | 61.251 | 61.655 | 1.0066 |
| 2 | 103.454 | 108.363 | 1.0474 |
| 4 | 166.000 | 170.901 | 1.0295 |

旧测量的吞吐比值为两并发 1.0474、四并发 1.0295，三档几何平均为 1.0277。由于实际计算分支相同，这些差异不能归因于优化，旧服务验收标记已撤销。正式启用决策以重新对照为准。

候选就绪时额外已分配 HBM 为 0；实际 KV 分配、四路完整上下文容量、MTP=2 与三档图捕获均保持。后台实验结束后另行恢复并核验正式 8100 服务。

## 算子复核

用两次独立进程与不同随机种子，分别测真实层 0/30/63、gate_up/down、M=6/12，共 24 个复核项。随机输入、零输入、64 倍幅值和图重放全部通过；最大相对 L2 误差为 0.001068。

主计时采用每张图 512 次调用、五轮交替配对 NPU 事件跨度并除以调用数，保留全部原始时间。此前一次 16 调用的复核与主机扩展编译重叠，出现明显抖动；该记录保留，未用于收益确认。两次 512 调用复核期间没有并行编译。

| M | 三层、两次复核的耗时下降范围 |
|---:|---:|
| 6 | 4.5%～11.7% |
| 12 | 20.3%～24.7% |

计算模式会改变有限精度的中间计算，误差通过 BF16 容差验证，不承诺逐位一致或生成 token 序列完全相同。MTP 接受率与步骤数保留在服务原始记录中；算子耗时下降不能直接当成服务 TPS 提升。

## 自定义内核改动

- `VectorNdV3`：将 group128 INT4 直接向量化反量化为 ND 面板，去掉 UB Gather；由 Cube 的 ND→zN 搬运完成格式转换。
- `panel_v3`：完整 K 权重面板反量化一次，供同一核心的全部 M tile 复用；测试更大的 M/N tile。
- `stream_nd_v3`：ND 双缓冲 K 流水，FP32 L0C 跨完整 K 累加，测试 M=256/N=128 与 M=128/N=256。

完整 K 面板版 28 个配置通过数值和图重放；滚动 ND 版 8 个大 M 配置通过，滚动版随机输入输出与原生的相对 L2 误差为 0。两种自定义路径仍慢于原生，保留为实验代码，没有安装到候选整模型。

INT8 投影的 FP32 尺度缓存复用、FP16 输出分支，以及更大 M 的 inner_precise=1 均做了短测，没有发现值得接入的稳定收益。

## 使用与复现

当前代码的诊断入口已复用主程序修复后的实现，可显式启用：

```bash
source scripts/env_ascend_291.sh
poetry run python -m tests.performance.qwen38.cli profile \
  --catalog configs/models.json --model-name qwen3.8-27b-optimized-v1 \
  --kv-cache-gib 22.5 --w4a8-linear-backend native-tuned \
  --cases generation512-c1,generation512-c2,generation512-c4 \
  --rounds 3 --no-capture --output-dir /absolute/new-output
```

该命令要求独占目标 NPU，默认不运行 GSM8K 或稳定性长测。算子入口为 `verify_w4_fastpath`，自定义内核入口为 `verify_w4_panel`；`--stream` 选择滚动 ND 版。汇总可通过 `summarize_native_tuned 实验目录` 重建。


## 收尾与验证

`poetry run pytest -q`：310 通过、13 跳过；后续 profiling/开关针对性检查 17 通过。`git diff --check` 通过。正式 8100 服务已恢复，真实 Responses 返回 `2`，NPU 上只有一个正式 Worker，18100 已释放。

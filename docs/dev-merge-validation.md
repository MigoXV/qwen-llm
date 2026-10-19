# dev 合并与默认模型回归

2026-10-16，当前分支 `feature/qwen-llm`。

## 合并结果

- 合并前当前分支：`8d40dde05815e78afc05a8521dbea39a8e3e56a5`。
- 合入远端 dev：`08b9ea551b96257d32ade7faa23a4536b8fb17a3`。
- 双亲合并提交：`95086734525ecb0fd3898ab7e5e5d87d770f52f0`。
- 本地备份引用：`backup/pre-dev-merge-20261016`。

远端dev的文件树与当前分支祖先`f949cb6`完全相同。确认其内容已包含后，
通过`git merge --no-ff -s ours origin/dev`补齐历史关系。
合并前后文件树完全一致，dev已成为当前分支祖先；当前27B及其他模型实现均保留。
没有更新远端分支。后续dev没有新提交时，可以快进到验收完成后的当前分支。

## 已完成的代码检查

- Poetry配置检查通过。
- 现有单元测试67项通过，耗时15.07秒；3条已有依赖弃用警告。
- 清除CANN注入的`PYTHONPATH`后，Poetry运行环境的`pip check`通过。
- 默认启动、模型启动、环境启动和量化基准脚本的Bash语法检查通过。
- mock拦截进程启动，验证默认CLI生成实际ServingConfig：MTP2、batch8192、
  KV9.375 GiB、FULL_DECODE_ONLY、异步调度、NPUGraphEx、前缀缓存、upstream后端，
  输入8192、输出8192、四路并发。
- 用临时Poetry替身验证默认脚本路由、仅传端口覆盖、显式选择模型；未加载模型。

原始配置与路由检查记录：
[/home/qwen-models/reports/qwen3.8-27b/dev-merge-9508673/code-checks.json](/home/qwen-models/reports/qwen3.8-27b/dev-merge-9508673/code-checks.json)。

## 已完成的默认配置实机回归

用户通知可停止小模型并开始测试后，检查发现小模型已退出、NPU无进程。
通过临时systemd单元调用默认`bash scripts/run_ascend.sh`，测量代码为`4f4a642`。
Worker确认两步MTP、AscendEagleProposer、3个图、完整上下文容量4.0路，
实际KV分配9,999,990,784字节，upstream W4后端；所有默认参数与历史配置一致。

执行期间用户要求简化并尽快结束。因此保留已完成的能力/检索检查及正在运行的
短输入基准，跳过长输入性能扫描和额外的实机限制/取消检查，只追加简短SDK冒烟。

| 检查 | 结果 |
| --- | --- |
| 原有72题能力检查 | 72/72，执行错误0；历史71/72 |
| 四路8K重复材料检索 | 4/4 |
| 四路8K混合中文检索 | 4/4 |
| OpenAI SDK严格校验、非流式和SSE | 正常，分别回答2和好 |
| 8并发过载检查 | 三轮共12个429，符合四路活动上限 |
| 正常退出 | SIGTERM后应用正常关闭，8100端口关闭，NPU无残留进程 |

能力检查差异属于这次输出结果；合并代码树未变，不把多答对一题归因为合并。

短输入沿用原始提示与128输出token，各并发三轮取中位数，四路为合计输出吞吐。
1/2/4路所有测量请求均正常生成128 tokens。测速过程为独占设备、同一服务实例。

| 并发 | 合并前TPS | 合并后TPS | 变化 |
| --- | ---: | ---: | ---: |
| 1 | 47.87 | 46.95 | -1.91% |
| 2 | 75.24 | 73.27 | -2.61% |
| 4 | 130.00 | 127.66 | -1.80% |

整卡HBM采样峰值41,215 MiB（40.249 GiB），与此前约40.25 GiB相当。
采样约每秒一次，另有命令执行开销，共250次，覆盖加载后半程及测试过程。
这是设备采样峰值，不是分配器瞬时峰值。

原始记录位于
[/home/qwen-models/reports/qwen3.8-27b/dev-merge-9508673/runtime](/home/qwen-models/reports/qwen3.8-27b/dev-merge-9508673/runtime)：
`quick-summary.json`为最终汇总，`short.json`保留逐轮结果，
`quality.json`与两份`retrieval*.json`保留逐题回答，`sdk-smoke.json`记录接口结果，
`npu-samples.jsonl`、`service-final.log`和`lifecycle.json`保留显存与生命周期证据。
原完整回归编排进程按用户缩减范围要求主动终止；其143退出码不表示模型故障，
最终结论以`quick-summary.json`为准。

当前结论：合并、代码检查和默认27B MTP2快速实机回归通过，未发现明显性能退化。
LLM和ASR系统服务保持测试前的inactive状态。其他模型与ASR未做实机复测。

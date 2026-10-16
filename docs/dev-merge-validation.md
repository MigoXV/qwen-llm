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

## 待执行的实机回归

用户正在运行另一小模型，明确要求先完成代码检查，由用户停止该任务后再通知启动。
本轮未启动LLM、未停止其他工作区进程、未运行实机能力或吞吐测试。
LLM和ASR系统服务检查时均为inactive。

收到设备释放通知后，仅启动默认`qwen3.8-27b-mtp2`，执行真实HTTP/SDK与SSE、
请求限制和取消、72题能力检查、两类8K检索、预热后的1/2/4路三轮测速、
长输入稳态测速、显存记录及正常退出检查。
对照历史71/72、单路47.87 TPS、四路130.00 TPS和40.25 GiB显存；
吞吐下降超过20%时追加三轮确认并定位。其他模型与ASR不做实机复测。

当前结论仅为合并及代码检查通过，实机回归尚未完成。

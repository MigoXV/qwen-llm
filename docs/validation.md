# W8A8 下载与 NPU 实测

日期：2026-10-15；工作分支 `feature/qwen-llm`，实现按功能分批提交，具体提交见 `git log`。原始日志及逐请求 JSON 位于 `/home/qwen-llm-build/logs`。

## 软件与硬件

| 项目 | 实测版本或配置 |
| --- | --- |
| 架构 | aarch64 |
| NPU | 物理设备 4，910B4-1，64GiB HBM |
| CANN | 8.3.RC1，未修改 |
| Python | 3.10.22，独立 Poetry 环境 |
| vLLM | 0.11.0+empty，从源码安装 |
| vLLM 源码 revision | b8b302cde434df8c9289a2b465406b47ebab1c2d |
| vllm-ascend | 0.11.0rc1，安装官方 wheel |
| 插件参考源码 revision | c5fe179cef7f91096c5090bdd3de846cab92b736 |
| Torch / Torch NPU | 2.7.1 / 2.7.1 |
| Transformers / Hugging Face Hub | 4.57.1 / 0.36.0 |
| setuptools | 79.0.1，保留历史 TorchAir 所需 pkg_resources |
| SDK 兼容性测试 | OpenAI Python 3.31.0，启用严格响应校验 |


## 模型与运行环境

下载来源：[aleoyang/Qwen3-14B-w8a8-MindIE](https://modelscope.cn/models/aleoyang/Qwen3-14B-w8a8-MindIE)。模型卡描述为 910B3 上的 MindIE W8A8 量化；本机在 910B4-1 上用 vLLM Ascend 实测通过。不是 GPU AWQ，也不是 310 专用 W8A8SC。

项目路径 `model-bin/Qwen3-14B-W8A8-Ascend` 通过符号链接指向 `/home/qwen-llm-build/models/Qwen3-14B-W8A8-Ascend`。下载约 19,924,499,492 字节，五个权重分片及所有配置按 ModelScope 每文件 revision 固定下载，大小与 SHA256 全部校验；完整清单在模型目录 `.modelscope-manifest.json`。未修改权重或量化配置。

实际 Poetry 环境为 `/home/qwen-llm-build/virtualenvs/qwen-llm-INLN9vSh-py3.10`。原 `/root/.cache/pypoetry/virtualenvs/...` 环境保留。源码、虚拟环境、模型、缓存和日志使用 `/home`，避免根分区耗尽；原 Poetry artifacts/cache 迁移后在原路径保留链接。

`torch_npu` 已验证 is_available=True、device_count=1，FP16 矩阵乘法实际运行在 npu:0。当前物理设备 4 映射为逻辑 0。服务使用 V1 AsyncLLM、单模型 Worker，量化后端 `ascend`，计算 dtype BF16。CANN AddRmsNormQuantV2 不接受本权重对应的 FP16 zeroPoints 输入，BF16 正常，因此保留融合算子并使用 BF16。

启动脚本加载 CANN 和与 Torch C++ ABI 匹配的 ATB，解决缺少 libatb.so 的依赖问题；没有修改厂商源码或系统软件。

## 显存与容量

相同 BF16/W8A8、2048-token 分块预填充、四路、输入/输出各8192上限。KV 理论最低10GiB，实际配置10.25GiB。

旧插件 NPUWorker 忽略上游 KV 显式预算，首次启动按整卡可用空间分配，产生231168个KV tokens、整卡59345MiB占用。项目子类在原始内存profile后应用明确预算，容量超出可用空间时报错；修复后为67072个KV tokens、524个128-token blocks，16384-token请求最大并发4.09。

| 配置 | 整卡 HBM（MiB） | 说明 |
| --- | ---: | --- |
| 无模型 | 3404 | 设备基础占用 |
| 旧插件未限KV | 59345 | 不是交付配置 |
| Eager，KV10.25GiB | 33695 | 模型进程30345MiB |
| PIECEWISE Graph，KV10.25GiB | 33885 | 模型进程30535MiB |

模型加载日志18.6374GB；KV预算不等于整卡占用。Graph捕获日志用时2秒、额外0.04GiB；其余额外占用还包含运行时和编译资源。1秒间隔采样见 `npu-samples.jsonl`，按phase区分启动与不同负载。初版采样器的结构化hbm_mib字段为空，已修正；历史采样从保留的原始npu-smi文本重新解析为 `npu-summary.json`。短测峰值Eager33847MiB、PIECEWISE33887MiB；长输入峰值分别34115/34133MiB（约33.3GiB）。

## 受控短输入基准

设备只有一个VLLMEngineCor进程；ASR已停止，无其他NPU推理负载。请求通过真实HTTP SSE，temperature=0，关闭思考，前缀缓存关闭。预热后每档三轮；表内吞吐为每轮总输出tokens/该轮总耗时的中位数。输入为“请连续列出至少100个常见汉语成语，每个成语附一句简短解释，不要省略，不要提前结束。”，输出上限128tokens。

| 并发 | Eager tokens/s | PIECEWISE tokens/s | Eager首token中位数 | PIECEWISE首token中位数 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 15.20 | 45.95 | 79ms | 62ms |
| 2 | 27.91 | 85.81 | 121ms | 101ms |
| 4 | 55.54 | 171.02 | 170ms | 138ms |
| 8次突发 | 56.41 | 171.76 | 167ms | 138ms |

8次突发每轮4个成功、4个429；吞吐只计算被接收的四路，不表示支持8路。原始 `benchmark-eager.json`、`benchmark-aclgraph.json` 含逐请求数据及P50/P95/P99。此结果代表当前短输入测试，不代表长上下文速度。

## 长输入与正确性

使用本地tokenizer构造四份模板后精确8192-token输入，开头不同，防止四路共享大段前缀。每档三轮，输出预算128。完整原始数据 `long-eager-128.json`、`long-aclgraph-128.json`。所有请求成功，cached_tokens=0。

Eager各轮总耗时13.54/12.54/12.51秒；PIECEWISE为8.17/7.91/7.97秒。部分请求自然EOS，且不同执行策略或轮次可能改变尾部生成，因此不能把这一组吞吐作为完全等长输出的严格加速比。可重放：

```bash
QWEN_LONG_TAG=check poetry run python -m tests.performance.qwen38.benchmark_long_context
```

该脚本默认三轮，QWEN_LONG_OUTPUT可调整输出预算；不设置ignore_eos，不承诺强制生成到预算。四路各8192输入已实际完成，四路同时生成满8192输出尚未实测；容量4.09来自实际KV块数。

三条固定短问题（自我介绍、123×7、键值缓存解释）在eager与PIECEWISE下文本及输出token数一致，见 `smoke-w8a8-eager.json`、`smoke-aclgraph.json`。数学结果861正确；这只是冒烟对齐，不是量化精度评估。真实思考SSE正确返回reasoning增量；512-token测试达到上限，最终response.incomplete，不能报告成完整数学推理通过。

## 工程验证与限制

41项测试通过，覆盖协议、严格OpenAI SDK解析、思考分块、token边界、四路准入、过载、超时、断连、SIGINT/SIGTERM、端口和资源清理、下载SHA256、量化识别、KV预算和NPU进程表解析。`poetry check`、`bash -n`、`compileall`、`git diff --check`另行执行。Poetry脚本声明及Starlette的AnyIO别名存在弃用提示，不影响通过。

原始AWQ模型真实直接启动被vLLM拒绝：`awq quantization is currently not supported in npu`，日志 `awq-real-start.log`。这解释了改用用户授权下载的Ascend W8A8权重；没有静默解量化或回退V0。

API监听127.0.0.1:8100；原8000服务未动。`qwen-asr-ascend.service`按授权停止、仍保留enabled。恢复ASR前先关闭本项目模型。

## 未采用的完整解码图

另行实际试验FULL_DECODE_ONLY，捕获1/2/4批次成功，但首个1-token warmup在ACL Graph replay处超过一分钟仍无响应，NPU AICore利用率为0，服务未就绪。发送SIGTERM后仍未退出，故仅对本次实验的API/Core执行强制停止，并清理其辅助进程；设备恢复3404MiB且无模型进程。未将此模式暴露为生产选项。日志 `serve-w8a8-full-decode.log`；不能宣称它已启用或有性能收益。

## 权重预取与前缀缓存

另测DENSE_OPTIMIZE=1、PREFETCH_MLP=1配合PIECEWISE和前缀缓存。相同短输入的所有请求cached_tokens均为0，单路45.38、四路168.21tokens/s，略低于未启用预取时45.95/171.02；首token延迟分别85/181ms，也高于62/138ms。因此最终关闭这两个预取相关开关。原始 `benchmark-optimized.json`，不将这个小幅差异宣称为确定性退化幅度。

同实例四路8K输入重放，首轮无缓存8.22秒，后两轮各约3.98秒，每路明确命中8064tokens。输出长度有所不同，主要证据是缓存命中数与首token延迟，不能只用总吞吐宣称精确倍数。数据 `long-optimized-128.json`。最终配置保留前缀缓存，并在未启用预取的新进程再次验证。

交付默认：Ascend W8A8、BF16、PIECEWISE ACL Graph（1/2/4批次）、融合RMSNorm+量化、分块预填充2048、前缀缓存、四路上限、10.25GiB KV。未启用Inductor、实验完整解码图或没有收益的MLP预取；不适用的MoE/MLA/多卡通信开关保持关闭。

## 最终实例复核

未启用预取的新进程再次通过四路精确8192-token输入（三轮）：第1轮：总耗时8.28秒，首token中位延迟3.049秒；第2轮：总耗时4.12秒，首token中位延迟0.239秒；第3轮：总耗时4.11秒，首token中位延迟0.236秒。首轮cached_tokens=0，后两轮每路8064。固定三条短问题文本和token数仍与eager一致，真实思考SSE按512-token预算返回合法incomplete。记录 `long-final-128.json`、`smoke-final.json`。

最终API进程2087639，Engine Core 2088757，仅一个NPU模型进程；就绪实例ID `aa6e325310ba4119bf87a15da03b13f9`。运行配置、进程、环境开关及设备盘点保存于 `deployment.json`。采样器已停止，模型服务保持运行。停止命令 `kill -TERM $(cat /home/qwen-llm-build/server.pid)`；重新启动 `bash scripts/run_ascend.sh`。

最终配置长输入测试整卡HBM峰值34171MiB（约33.4GiB），包含设备基础占用，见 `npu-summary.json` 的final-long-128项。

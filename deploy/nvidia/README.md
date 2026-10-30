# NVIDIA 环境补丁

`runtime.txt` 和 `quant.txt` 分别固定推理、量化环境。`scripts/install_nvidia.sh` 仅在 runtime 环境中执行 `poetry run python scripts/patch_nvidia.py`，不修改 Ascend 或量化工具环境。

## vLLM 0.32.0 的 SHM 通知积压

`SpinCondition` 使用共享内存传输数据，使用 ZeroMQ SUB/PUB 发送唤醒通知。读者忙循环时只读共享内存，长期不访问通知 socket。`CONFLATE=1` 本身不足以释放这种情况下积累的通知；ZeroMQ 文档要求调用 `recv` 或查询 `ZMQ_EVENTS` 才触发合并清理。[官方说明](https://libzmq.readthedocs.io/en/latest/zmq_setsockopt.html#zmq-conflate-keep-only-last-message)

`patches/shm-notify-conflate.patch` 在读者线程每成功读取 256 条 SHM 消息后，查询一次通知 socket 的 `ZMQ_EVENTS`。它不消费取消信号，不改变模型算法、缓存精度、调度或 SHM 写入标记，也不增加后台线程或每请求 GC。

补丁只接受 `shm-notify-conflate.json` 指定的 vLLM 版本与原始文件 SHA256；重复运行不重复修改，源码未知时拒绝覆盖。已有 runtime 可在停止实例后执行：

```bash
export QWEN_NVIDIA_ENV=runtime
source scripts/env_nvidia.sh
poetry run python scripts/patch_nvidia.py
```

修改后重启实例生效。包版本号仍为 0.32.0，复现实验须同时记录补丁清单；仅用 `pip freeze` 无法表示源码补丁。

## Qwen3.5 词嵌入量化接线

安装版本已实现 compressed-tensors 的 INT4/INT8 词嵌入查表反量化，但 `Qwen3_5Model` 构造 `VocabParallelEmbedding` 时没有传递 `quant_config` 和层名前缀。词嵌入量化产物因此无法加载 `embed_tokens.weight_packed`。

`patches/qwen35-embedding-quant.patch` 只补齐这两个构造参数，使用 vLLM 已有查表内核。量化方案由产物 `config.json` 决定，没有匹配词嵌入的量化方案时仍使用原浮点权重。补丁同样受版本和前后 SHA256 约束；只在 NVIDIA runtime 应用。

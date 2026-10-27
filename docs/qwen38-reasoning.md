# Qwen3.8-27B 原生思考档位

## 官方能力与问题核实

[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)和[聊天模板](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/chat_template.jinja)支持 `low`、`medium`、`xhigh` 三个思考档位。原始模板默认启用思考并使用 `xhigh`；本服务沿用默认关闭思考的接口行为。`high` 是服务兼容别名，不是原始模型的第四个思考档位。

本地原始 BF16、原混合量化版及 `GDNHeadW8-v1` 优化版的 `chat_template.jinja` 完全一致，SHA256 为 `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`。原始下载清单记录该模板的 ModelScope revision 为 `a23c845fc2aa3b7f3979e32397cea12a57551631`。

原实现把请求中的 `low/medium/high` 转为 `enable_thinking=True`，没有传递 `reasoning_effort`。实际 tokenizer 验证显示，三个请求生成相同的 token IDs，均使用模板默认的 `xhigh`。此外，同时传递布尔 `true` 时，参数校验会把明确指定的档位重写为 `medium`，但运行时仍落到 `xhigh`。

## 修复后的接口

```json
{
  "model": "qwen3.8-27b-optimized-v1",
  "input": "计算 17 × 23，并检查结果。",
  "reasoning": {"effort": "low"},
  "max_output_tokens": 8192
}
```

- `none` 关闭思考；`low/medium/xhigh` 直接传入原始模板；`high` 规范化为 `xhigh`，响应回显 `xhigh`。
- 没有指定思考参数时仍使用 `none`；仅传顶层或 `chat_template_kwargs` 中的 `enable_thinking=true` 时使用 `xhigh`。
- `chat_template_kwargs.reasoning_effort` 是另一个档位入口。重复指定的档位和开关必须一致，冲突返回 400，且不占用生成名额。
- `enable_thinking=true` 不覆盖明确指定的档位。结构化输出继续只支持关闭思考。
- 输出预算包含思考和正文；各档不设置额外硬性 token 预算，不要求每道题的思考长度严格递增。

内部 `Engine.prepare` 接收 `ReasoningEffort` 枚举；仅向支持原生档位的模板传递 `reasoning_effort`。旧模型保留布尔开关行为。`/health/ready` 的 `runtime.supports_reasoning_effort` 和 `runtime.reasoning_efforts` 用于区分两种能力。

## 回归验证

加载 `scripts/env_ascend_291.sh` 后执行：

```bash
poetry check
poetry run pytest -q
poetry run python -m tests.performance.qwen38.cli smoke --help
```

全量测试 **215 项通过**。新增覆盖档位传递、别名规范化、布尔兼容、重复配置与冲突、非法值、结构化输出限制、HTTP/SSE 及 SDK。三个本地模型的实际 tokenizer 均与原始模板逐 token 对齐，覆盖四档、系统指令和多轮历史。测试不加载模型权重；未安装本地 tokenizer 的环境跳过本地对齐用例，不自动下载。

## 真实 NPU 验收

2026-10-23 在现有 `qwen-llm.service` 上重启并验证 `qwen3.8-27b-optimized-v1`，19 个场景全部通过：四档非流式生成、四路混合档位 SSE、默认关闭思考、`high` 别名、仅布尔开关、兼容参数保留 `low`、四档单 token 截断，以及三种参数错误。每个有效请求均核对实际输入 token 数与对应原始模板展开结果，检查思考/正文分离及 usage；并发请求使用独立结果标记验证隔离。

| 单请求档位 | 输入 token | 总输出 token | 思考 token |
| --- | ---: | ---: | ---: |
| none | 35 | 20 | 0 |
| low | 63 | 118 | 97 |
| medium | 33 | 151 | 105 |
| xhigh | 76 | 138 | 116 |

输入为 `17 × 23` 和各请求的独立结果标记，均得到正确结果 391。此表为单次功能冒烟记录，不作为各档质量或性能排名。单 token 输出限制均返回 `incomplete/max_output_tokens`；参数错误均返回 400，服务保持就绪，最终活动请求数为 0。

验收前后为同一实例 `e907937e6f554f838a71f45d0c4aa001`。Worker 仍启用 MTP2、三组解码图，完整上下文容量约 4.036 路。源码哈希、模板哈希、完整请求、HTTP/SSE 原始响应、token 用量、验收脚本及前后健康信息保存在：

```text
/home/qwen-models/reports/qwen3.8-27b/reasoning-20261023T084335Z/
```

`summary.json` 汇总各项结果，`verify.py` 可在同一 Poetry 环境重新执行并创建新的带时间戳结果目录。

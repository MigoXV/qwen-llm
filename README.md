# qwen-llm

Qwen LLM 的 Python 工程骨架。目前提供版本查询入口，尚未实现模型加载或推理。

## 环境安装

使用 Python 3.10 和 Poetry 管理环境、依赖及运行命令：

```bash
poetry env use python3.10
poetry install
```

依赖使用清华 PyPI 镜像，锁定版本记录在 `poetry.lock` 中。

## 命令行

```bash
poetry run qwen-llm --help
poetry run qwen-llm version
poetry run python -m qwen_llm.commands.app version
```

CLI 入口位于 `src/qwen_llm/commands/app.py`，通过 Typer 注册命令，并统一配置基础日志。

## 开发与调试

在 VS Code 中选择 Poetry 虚拟环境解释器，再使用 `app` 配置调试模块入口。
调试配置从项目根目录的 `.env` 读取环境变量；该文件可按需自行创建，不提交到 Git。

```bash
poetry check
poetry run pytest
```

`tests/` 为预留测试目录，目前没有业务测试。模型文件放在 `model-bin/`，数据及输出可分别放在 `data-bin/`、`outputs/`，这些目录均被 Git 忽略。

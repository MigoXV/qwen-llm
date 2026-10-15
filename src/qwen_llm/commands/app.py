from __future__ import annotations

import logging
from importlib.metadata import version

import typer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Qwen LLM 命令行工具。", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Qwen LLM 项目工具。"""


@app.command("version")
def show_version() -> None:
    """显示已安装的项目版本。"""
    typer.echo(version("qwen-llm"))


if __name__ == "__main__":
    app()

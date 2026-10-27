"""四路8192-token材料检索，检查长上下文和并发请求隔离。"""
import asyncio
import json
from pathlib import Path
import time
from urllib.parse import urlsplit

import httpx
import typer

from tests.performance.qwen38.evaluation import check_answer
from qwen_llm.engine.tokenizer import tokenizer_path

app = typer.Typer()


@app.command()
def main(
    model_path: Path = typer.Option(..., envvar="QWEN_MODEL_PATH"),
    model_name: str = typer.Option(..., envvar="QWEN_MODEL_NAME"),
    output: Path = typer.Option(..., envvar="QWEN_EVAL_LONG_OUTPUT"),
    base_url: str = typer.Option("http://127.0.0.1:8100/v1", envvar="QWEN_BASE_URL"),
    material: str = typer.Option("repeated", envvar="QWEN_EVAL_LONG_MATERIAL"),
):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path(model_path), local_files_only=True)
    if material not in {"repeated", "mixed"}:
        raise typer.BadParameter("material须为repeated或mixed")
    topics = [
        "城市图书馆根据主题和出版年份整理图书，读者通过预约系统办理借阅。",
        "气象站按小时记录温度、风速和降雨量，并在每天傍晚核查观测表。",
        "自然保护区的工作人员沿着固定路线观察鸟类，记录植物和水源分布。",
        "学校实验室为水样测量酸碱度，实验结束后整理仪器和测量记录。",
        "维修人员根据巡检单检查车间传感器，在交班时说明需要更换的部件。",
        "交通部门统计公交线路的客流，比较工作日与周末的运行情况。",
        "社区中心组织志愿活动，为参与者登记时间并准备活动材料。",
        "农业试验小组比较不同灌溉方式，每周测量土壤湿度和作物高度。",
        "软件团队按模块检查构建日志，在发布前核对配置与使用说明。",
        "档案馆为历史照片补充拍摄时间和地点，再按照保存要求完成归档。",
    ]
    body = "\n".join(f"记录{j+1}：{topics[j % len(topics)]}" for j in range(500))
    body_ids = tokenizer.encode(body, add_special_tokens=False) if material == "mixed" else None
    items = []
    for i, code in enumerate(["AZ17B9", "CY83D2", "EX49F6", "GV25H1"]):
        prefix = f"独立材料{i}。本材料的唯一校验码是{code}。以下是无关填充：\n"
        suffix = "\n材料结束。只输出本材料的校验码，不要解释或添加其他文字。"
        def tokens(content):
            return len(tokenizer.apply_chat_template([{"role": "user", "content": content}],
                       tokenize=True, add_generation_prompt=True, enable_thinking=False))
        count = 8192 - tokens(prefix + suffix)
        for n in range(max(0, count-32), count+33):
            filler = " data" * n if body_ids is None else tokenizer.decode(body_ids[:n])
            if "\ufffd" in filler:
                continue
            prompt = prefix + filler + suffix
            if tokens(prompt) == 8192:
                items.append({"id": i, "prompt": prompt, "expected": code, "check": "exact"})
                break
        else:
            raise RuntimeError("无法构造8192-token材料")

    async def run():
        origin = urlsplit(base_url)
        health_url = f"{origin.scheme}://{origin.netloc}/health/ready"
        async with httpx.AsyncClient(timeout=650) as client:
            health = await client.get(health_url)
            health.raise_for_status()
            service = health.json()
            if service["model"] != model_name:
                raise ValueError("检索验证与服务模型身份不一致")
            async def one(case):
                start = time.perf_counter()
                try:
                    r = await client.post(base_url.rstrip("/") + "/responses", json={
                        "model": model_name, "input": case["prompt"], "temperature": 0,
                        "reasoning": {"effort": "none"}, "max_output_tokens": 64})
                    r.raise_for_status()
                    value = r.json()
                    answer = "".join(part.get("text", "") for item in value.get("output", [])
                                     if item.get("type") == "message" for part in item.get("content", []))
                    return {**case, "response": value, "text": answer, "seconds": time.perf_counter()-start,
                            "passed": value.get("status") == "completed" and check_answer(case, answer)}
                except Exception as exc:
                    return {**case, "error": str(exc), "passed": False, "seconds": time.perf_counter()-start}
            samples = await asyncio.gather(*(one(case) for case in items))
            final = await client.get(health_url)
            final.raise_for_status()
            record = {"service": service, "material": material, "input_tokens": 8192, "concurrency": 4, "samples": samples,
                      "passed": sum(s["passed"] for s in samples), "total": 4,
                      "instance_unchanged": final.json()["instance_id"] == service["instance_id"]}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        typer.echo(f"Long-context retrieval: {record['passed']}/4")
        if record["passed"] != 4 or not record["instance_unchanged"]:
            raise typer.Exit(1)
    asyncio.run(run())


if __name__ == "__main__":
    app()

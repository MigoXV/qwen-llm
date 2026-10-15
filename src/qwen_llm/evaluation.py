"""可复现的文本能力冒烟检查；不是通用能力榜单或完整量化精度评估。"""
import asyncio
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata
from urllib.parse import urlsplit

import httpx


def cases():
    items = []
    for a, b in [(17, 19), (23, 14), (37, 12), (48, 25), (64, 16),
                 (123, 7), (81, 11), (42, 13), (29, 31), (99, 9)]:
        items.append(("arithmetic", f"计算 {a} 乘以 {b}。只输出答案数字。", str(a * b), "exact"))
    for a, b in [(127, 38), (999, 126), (504, 237), (1001, 519), (80, 123)]:
        items.append(("arithmetic", f"计算 {a} 减去 {b}。只输出答案数字。", str(a - b), "exact"))
    for a, b in [(144, 12), (225, 15), (1008, 24), (1728, 36), (729, 9)]:
        items.append(("arithmetic", f"计算 {a} 除以 {b}。只输出答案数字。", str(a // b), "exact"))
    for prompt, answer in [
        ("水的化学式是什么？只输出化学式。", "H2O"),
        ("法国的首都是什么？只输出中文城市名。", "巴黎"),
        ("Python 定义函数使用哪个关键字？只输出关键字。", "def"),
        ("最大的两位质数是多少？只输出数字。", "97"),
        ("二进制 1010 转为十进制是多少？只输出数字。", "10"),
        ("1 千米等于多少米？只输出数字。", "1000"),
        ("所有猫都是动物，小花是一只猫。小花是动物吗？只回答“是”或“否”。", "是"),
        ("甲比乙高，乙比丙高。甲和丙谁更高？只回答“甲”或“丙”。", "甲"),
        ("数列 2、4、8、16 的下一个数是多少？只输出数字。", "32"),
        ("一个盒子有5个红球和3个蓝球，共有几个球？只输出数字。", "8"),
    ]:
        items.append(("knowledge_logic", prompt, answer, "exact"))
    for source, target in [("苹果", "apple"), ("猫", "cat"), ("狗", "dog"),
                           ("水", "water"), ("红色", "red"), ("书", "book"),
                           ("太阳", "sun"), ("月亮", "moon")]:
        items.append(("translation", f"将“{source}”翻译成英语，只输出对应的英语单词。", target, "exact"))
    items.extend([
        ("format", '只输出一个JSON对象，其唯一键为x，值为整数1。不要输出代码块。', {"x": 1}, "json"),
        ("format", "依次输出三个英文字母A、B、C，用英文逗号分隔，不要空格或其他内容。", "A,B,C", "exact"),
    ])
    return [{"id": i, "category": c, "prompt": p, "expected": e, "check": k}
            for i, (c, p, e, k) in enumerate(items)]


def check_answer(case, answer):
    text = unicodedata.normalize("NFKC", answer).strip()
    if case["check"] == "json":
        try:
            value = json.loads(text)
            return value == case["expected"] and type(value.get("x")) is int
        except ValueError:
            return False
    if case["check"] == "final_number":
        match = re.search(r"(?:^|\n)FINAL:\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*$", text)
        try:
            return bool(match) and Decimal(match[1].replace(",", "")) == Decimal(case["expected"])
        except InvalidOperation:
            return False
    text = re.sub(r"[。.!！]+$", "", text).strip()
    return text.casefold() == str(case["expected"]).casefold()


async def evaluate(base_url: str, model: str, output: Path, math_dataset: Path | None = None,
                   math_samples: int = 32):
    semaphore = asyncio.Semaphore(4)
    items = cases()
    record = {"model": model, "created_at": datetime.now(timezone.utc).isoformat(),
              "kind": "deterministic correctness smoke with optional GSM8K subset",
              "limitation": "没有同模型浮点基线；小样本不能代表完整GSM8K、GPQA、代码或长推理能力。",
              "temperature": 0, "reasoning_effort": "none", "concurrency": 4, "samples": []}
    if math_dataset is not None:
        content = math_dataset.read_bytes()
        record["math_dataset"] = {"path": str(math_dataset.resolve()),
                                  "sha256": hashlib.sha256(content).hexdigest(), "selection": f"first {math_samples}"}
        for i, line in enumerate(content.decode().splitlines()[:math_samples]):
            row = json.loads(line)
            gold = row["answer"].rsplit("####", 1)[1].strip().replace(",", "")
            Decimal(gold)
            items.append({"id": f"gsm8k-{i}", "category": "gsm8k", "expected": gold,
                          "check": "final_number", "prompt": row["question"] +
                          "\nSolve the problem. End with a separate line FINAL: <answer number>."})
    record["cases_sha256"] = hashlib.sha256(json.dumps(items, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    async with httpx.AsyncClient(timeout=650) as client:
        origin = urlsplit(base_url)
        health_url = f"{origin.scheme}://{origin.netloc}/health/ready"
        health = await client.get(health_url)
        health.raise_for_status()
        record["service"] = health.json()
        if record["service"]["model"] != model:
            raise ValueError("能力评估模型与服务身份不一致")
        async def one(case):
            async with semaphore:
                started = time.perf_counter()
                try:
                    response = await client.post(base_url.rstrip("/") + "/responses", json={
                        "model": model, "input": case["prompt"], "temperature": 0,
                        "reasoning": {"effort": "none"},
                        "max_output_tokens": 2048 if case["category"] == "gsm8k" else 256})
                    response.raise_for_status()
                    value = response.json()
                    text = "".join(part.get("text", "") for item in value.get("output", [])
                                   if item.get("type") == "message" for part in item.get("content", []))
                    return {**case, "response": value, "text": text,
                            "passed": check_answer(case, text), "seconds": time.perf_counter() - started}
                except Exception as exc:
                    return {**case, "passed": False, "error": str(exc),
                            "seconds": time.perf_counter() - started}
        record["samples"] = await asyncio.gather(*(one(case) for case in items))
        final_health = await client.get(health_url)
        final_health.raise_for_status()
        record["instance_unchanged"] = final_health.json().get("instance_id") == record["service"].get("instance_id")
    record["passed"] = sum(row["passed"] for row in record["samples"])
    record["total"] = len(record["samples"])
    record["pass_rate"] = record["passed"] / record["total"]
    record["execution_errors"] = sum("error" in row for row in record["samples"])
    record["categories"] = {category: {"passed": sum(r["passed"] for r in record["samples"] if r["category"] == category),
                                       "total": sum(r["category"] == category for r in record["samples"])}
                            for category in sorted({r["category"] for r in items})}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    return record

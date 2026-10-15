"""从原始结果生成逐配置汇总；失败、缺失与缓存命中仍明确记录。"""
import json
from pathlib import Path
import re
import statistics

import typer

app = typer.Typer()


def summarize(directory: Path) -> dict:
    row = {"directory": str(directory), "parse_errors": []}

    def read(name):
        path = directory / name
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            row["parse_errors"].append({"file": name, "error": str(exc)})
            return None

    health = read("health.json")
    row["health"] = health
    exit_code = directory / "suite-exit-code.txt"
    row["suite_exit_code"] = int(exit_code.read_text()) if exit_code.exists() else None
    quality = read("quality.json")
    if quality:
        row["quality"] = {key: quality[key] for key in ("passed", "total", "execution_errors", "categories", "instance_unchanged")}
    for name in ("retrieval", "retrieval_mixed"):
        result = read(name + ".json")
        row[name] = ({key: result[key] for key in ("passed", "total", "instance_unchanged")}
                     if result else None)

    hbm = []

    def memory(resources):
        for used, total in re.findall(r"(\d+)\s*/\s*(\d+)\s*\|", resources.get("npu", "")):
            if int(total) >= 32768:
                hbm.append(int(used))

    short = read("short.json")
    if short:
        memory(short.get("before", {}))
        memory(short.get("after", {}))
        row["short"] = {}
        for concurrency in (1, 2, 4):
            rounds = [item for item in short["rounds"] if item["concurrency"] == concurrency]
            valid = short.get("valid", False) and len(rounds) == 3 and all(
                len(item["samples"]) == concurrency and all(
                    sample.get("http_status") == 200 and
                    sample.get("response", {}).get("status") in ("completed", "incomplete")
                    for sample in item["samples"]) for item in rounds)
            row["short"][str(concurrency)] = {
                "valid": valid,
                "round_count": len(rounds),
                "tps_median": statistics.median(item["tokens_per_second"] for item in rounds) if valid else None,
            }
        for item in short["rounds"]:
            memory(item.get("resources", {}))

    long = read("logs/long-steady-128.json")
    if long:
        rounds = long["rounds"]
        valid = long.get("after", {}).get("isolated", False) and len(rounds) >= 4 and all(
            item.get("resources", {}).get("isolated", False) and len(item["samples"]) == 4 and all(
            sample.get("response", {}).get("status") in ("completed", "incomplete")
            for sample in item["samples"]) for item in rounds)
        row["long"] = {
            "valid": valid, "round_count": len(rounds),
            "transition_rounds": max(0, len(rounds)-4),
            "cold_tps": rounds[0]["tokens_per_second"] if valid else None,
            "repeat_tps_median": statistics.median(item["tokens_per_second"] for item in rounds[-3:]) if valid else None,
            "repeat_cached_tokens": sorted({sample["response"]["usage"]["input_tokens_details"]["cached_tokens"]
                for item in rounds[-3:] for sample in item["samples"] if sample.get("response")}),
        }
        for item in rounds:
            memory(item.get("resources", {}))
    row["hbm_sampled_max_mib"] = max(hbm) if hbm else None
    return row


@app.command()
def main(
    report_root: Path = typer.Option(Path("/home/qwen-models/reports"), exists=True, file_okay=False),
    output: Path = typer.Option(Path("/home/qwen-models/reports/summary.json")),
):
    records = [summarize(path.parent) for path in sorted(report_root.glob("*/*/doctor.json"))]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"runs": records, "hbm_note": "整卡轮次采样最大值，非连续峰值"}, ensure_ascii=False, indent=2) + "\n")
    typer.echo(f"Summarized {len(records)} runs: {output}")


if __name__ == "__main__":
    app()

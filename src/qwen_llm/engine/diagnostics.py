"""无需创建模型或设备上下文的安装与兼容性检查。"""
import importlib.metadata
import json
import platform
import os
import subprocess
import sys
from pathlib import Path


def diagnose(model: Path, probe_runtime: bool = False) -> dict:
    packages = {}
    for name in ("vllm", "vllm-ascend", "torch", "torch-npu", "transformers", "huggingface-hub", "setuptools", "openai"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    blockers = []
    if (model / '.quantization.incomplete').exists():
        blockers.append('本地量化尚未完成，拒绝加载中间权重')
    config = {}
    model_quant_type = None
    weight_files = set()
    try:
        config = json.loads((model / "config.json").read_text())
    except (OSError, ValueError) as exc:
        blockers.append(f"模型配置不完整：{exc}")
    try:
        indexes = sorted(model.glob("*.safetensors.index.json"))
        if indexes:
            for index_path in indexes:
                index = json.loads(index_path.read_text())
                weight_files.update(index["weight_map"].values())
        else:
            weight_files.update(p.name for p in model.glob("*.safetensors"))
        if not weight_files:
            blockers.append("模型文件不完整：缺少 safetensors 权重或索引")
        for shard in weight_files:
            relative = Path(shard)
            if relative.is_absolute() or ".." in relative.parts:
                blockers.append(f"非法权重路径：{shard}")
            elif not (model / shard).is_file():
                blockers.append(f"缺少权重分片：{shard}")
        manifest = model / ".modelscope-manifest.json"
        if manifest.exists() and not json.loads(manifest.read_text()).get("verified"):
            blockers.append("ModelScope 下载尚未完成 SHA256 校验")
        local_manifest = model / 'quantization_manifest.json'
        if local_manifest.exists() and not json.loads(local_manifest.read_text()).get('export_complete'):
            blockers.append('本地量化清单尚未标记导出完成')
    except (OSError, ValueError, KeyError) as exc:
        blockers.append(f"模型文件不完整：{exc}")
    quant = config.get("quantization_config", {}).get("quant_method")
    description = model / "quant_model_description.json"
    if description.exists() and quant != "awq":
        try:
            model_quant_type = json.loads(description.read_text()).get("model_quant_type")
            if model_quant_type not in {"W8A8", "W8A8_DYNAMIC", "W4A8_DYNAMIC"}:
                blockers.append(f"尚未验证的昇腾量化类型：{model_quant_type}")
            quant = "ascend"
        except (OSError, ValueError) as exc:
            blockers.append(f"量化描述文件无效：{exc}")
    if quant == "awq":
        blockers.append("该 AWQ GEMM 模型未纳入已验证的 Ascend 配置，请选用模型目录中的 W8A8/W4A8 权重")
    if packages["vllm"] and packages["vllm"].startswith("0.11.") and (
        config.get("model_type") in {"qwen3_5", "qwen3_5_moe"} or quant == "compressed-tensors"
    ):
        blockers.append("该模型需使用新Ascend环境，请通过scripts/run_model.sh启动")
    expected = {"vllm": "0.11.0", "vllm-ascend": "0.11.0rc1",
                "torch": "2.7.1", "torch-npu": "2.7.1", "transformers": "4.57.1",
                "huggingface-hub": "0.36.0", "setuptools": "79.0.1", "openai": "3.31.0"}
    if packages["vllm"] and packages["vllm"].split("+")[0] == "0.18.0":
        expected.update({"vllm": "0.18.0", "vllm-ascend": "0.18.0",
                         "torch": "2.9.1", "torch-npu": "2.9.1", "transformers": "4.57.6"})
    for name, prefix in expected.items():
        if packages[name] and packages[name].split("+")[0] != prefix:
            blockers.append(f"{name} 版本 {packages[name]} 不匹配诊断基线 {prefix}")
    for name, value in packages.items():
        if value is None:
            blockers.append(f"未安装 {name}")
    cann_root = Path(os.getenv("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    cann = cann_root / "version.cfg"
    if not cann.exists():
        cann = cann_root / "share/info/runtime/version.info"
    devices = sorted(str(p) for p in Path("/dev").glob("davinci[0-9]*"))
    if not devices:
        blockers.append("未发现可见 Ascend NPU 设备节点")
    runtime_probe = None
    if probe_runtime:
        try:
            result = subprocess.run(
                [sys.executable, "-c", "import torch, torch_npu, vllm_ascend; from vllm.v1.engine.async_llm import AsyncLLM; assert torch.npu.is_available(), 'Ascend NPU is not visible'; print('V1 import OK; NPU', torch.npu.device_count(), torch.npu.get_device_name(0))"],
                env={**os.environ, "VLLM_USE_V1": "1"}, capture_output=True, text=True, timeout=45)
            runtime_probe = {"returncode": result.returncode, "stdout": result.stdout[-8000:], "stderr": result.stderr[-8000:]}
            if result.returncode:
                blockers.append("V1/Ascend 运行时导入失败，详见 runtime_probe")
        except subprocess.TimeoutExpired:
            runtime_probe = {"error": "45 秒导入超时"}
            blockers.append("V1/Ascend 运行时导入超时")
    return {"runtime_probe": runtime_probe, "devices": devices, "architecture": platform.machine(), "model": str(model.resolve()),
            "quantization": quant, "model_quant_type": model_quant_type,
            "weight_files": sorted(weight_files), "packages": packages,
            "cann_root": str(cann_root), "cann": cann.read_text() if cann.exists() else None,
            "blockers": blockers, "ready": not blockers}

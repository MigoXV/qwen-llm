"""CUDA 平台启动与观测；请求处理复用共享 AsyncGenerationEngine。"""
import asyncio
from dataclasses import dataclass
from importlib.metadata import version
import json
import os
from pathlib import Path

from qwen_llm.config import GraphMode, InferenceConfig, ReasoningEffort, Runner
from qwen_llm.engine.generation import AsyncGenerationEngine


@dataclass(frozen=True)
class NvidiaConfig(InferenceConfig):
    runner: Runner = Runner.EAGER
    graph_mode: GraphMode = GraphMode.FULL_DECODE_ONLY
    kv_cache_bytes: int | None = None
    gpu_memory_utilization: float = 0.9
    prefix_caching: bool = False
    tensor_parallel_size: int = 2
    kv_cache_dtype: str = 'auto'
    attention_backend: str | None = None
    experimental: bool = False
    vision: bool = False
    max_images_per_prompt: int = 1
    trim_history_images: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.tensor_parallel_size not in (1, 2):
            raise ValueError('本机 CUDA 配置只验证 TP1/TP2')
        if self.runner is Runner.ACLGRAPH:
            raise ValueError('CUDA 配置不能使用 ACL Graph')
        if self.kv_cache_dtype not in ('auto', 'fp8_e4m3'):
            raise ValueError('KV dtype 只允许 auto 或实验 fp8_e4m3')
        if self.kv_cache_dtype != 'auto' and not self.experimental:
            raise ValueError('FP8 KV 仅能由独立实验入口启用')
        if self.attention_backend not in (None, 'FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN'):
            raise ValueError('不支持的实验 attention backend')
        if self.attention_backend is not None and not self.experimental:
            raise ValueError('固定 attention backend 仅用于独立对照实验')
        if self.vision and not self.experimental:
            raise ValueError('视觉输入目前仅供实验性入口使用')
        if self.max_images_per_prompt < 1:
            raise ValueError('每请求图像数量须至少为 1')


def inspect_model(model: Path) -> dict:
    if (model / '.quantization.incomplete').exists():
        raise ValueError('量化尚未完成')
    manifest = model / 'quantization_manifest.json'
    if manifest.exists() and not json.loads(manifest.read_text()).get('export_complete'):
        raise ValueError('量化导出清单尚未完成')
    if (model / 'quant_model_description.json').exists():
        raise ValueError('CUDA 入口不能加载 Ascend 专用量化产物')
    config = json.loads((model / 'config.json').read_text())
    index = json.loads((model / 'model.safetensors.index.json').read_text())
    for name in set(index['weight_map'].values()):
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or not (model / relative).is_file():
            raise ValueError(f'非法或缺失的权重分片：{name}')
    return config


def has_visual_weights(model: Path) -> bool:
    index = json.loads((model / 'model.safetensors.index.json').read_text())
    return any('visual.' in name for name in index['weight_map'])


class NvidiaEngine(AsyncGenerationEngine):
    def metrics_text(self):
        from prometheus_client import generate_latest
        return generate_latest()

    async def start(self):
        config = self.config
        model_config = inspect_model(config.model_path)
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != config.tensor_parallel_size:
            raise ValueError('CUDA_VISIBLE_DEVICES 必须恰好对应 TP 使用的设备')
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        from transformers import AutoTokenizer
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        self.tokenizer = await asyncio.to_thread(
            AutoTokenizer.from_pretrained, str(config.model_path), local_files_only=True)
        template = self.tokenizer.get_chat_template()
        self.supports_thinking = 'enable_thinking' in template
        self.supports_reasoning_effort = 'reasoning_effort' in template
        self.supports_tools = all(marker in template for marker in (
            '<tool_call>', '<function=', '<parameter=', 'message.tool_calls', '<tool_call>'))
        if config.vision:
            vision_start = model_config.get('vision_start_token_id')
            image_pad = model_config.get('image_token_id')
            vision_end = model_config.get('vision_end_token_id')
            if vision_start is None or image_pad is None or vision_end is None:
                raise ValueError('模型 config.json 缺少 vision_start/image/vision_end token id，无法接线视觉输入')
            if not has_visual_weights(config.model_path):
                raise ValueError('模型工件缺少视觉权重（safetensors 中没有 visual.* 条目），不能启用视觉输入')
            self.image_pad_token_id = image_pad
            self.image_placeholder = self.tokenizer.decode([vision_start, image_pad, vision_end], skip_special_tokens=False)
        options = {}
        if config.runner is Runner.CUDAGRAPH:
            options['compilation_config'] = {'cudagraph_mode': config.graph_mode.value}
        if config.speculative_tokens:
            options['speculative_config'] = {'method': 'qwen3_5_mtp',
                'num_speculative_tokens': config.speculative_tokens}
        if config.async_scheduling is not None:
            options['async_scheduling'] = config.async_scheduling
        if config.attention_backend is not None:
            options['attention_backend'] = config.attention_backend
        if config.vision:
            options['limit_mm_per_prompt'] = {'image': config.max_images_per_prompt}
        args = AsyncEngineArgs(
            model=str(config.model_path.resolve()), tokenizer=str(config.model_path.resolve()),
            dtype=config.dtype.value, language_model_only=not config.vision,
            tensor_parallel_size=config.tensor_parallel_size, distributed_executor_backend='mp',
            disable_custom_all_reduce=os.getenv('NCCL_P2P_DISABLE') == '1',
            worker_cls='qwen_llm.engine.nvidia_worker.ObservedGPUWorker',
            max_model_len=config.max_input_tokens + config.max_output_tokens,
            max_num_seqs=config.max_concurrency, max_num_batched_tokens=config.batch_tokens,
            enforce_eager=config.runner is Runner.EAGER,
            enable_chunked_prefill=True, enable_prefix_caching=config.prefix_caching,
            mamba_cache_mode='align' if config.prefix_caching else 'none',
            kv_cache_memory_bytes=config.kv_cache_bytes, kv_cache_dtype=config.kv_cache_dtype,
            gpu_memory_utilization=config.gpu_memory_utilization, disable_log_stats=False, **options,
        )
        self.llm = AsyncLLM.from_engine_args(args)
        await self.health()
        async for _ in self.generate([self.tokenizer.eos_token_id], 'warmup', max_tokens=1):
            pass
        self.runtime_identity = {
            'platform': 'cuda', 'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
            'nccl_p2p_disabled': os.getenv('NCCL_P2P_DISABLE') == '1',
            'tensor_parallel_size': config.tensor_parallel_size,
            'packages': {name: version(name) for name in ('torch', 'vllm', 'transformers', 'compressed-tensors')},
            'quantization': model_config.get('quantization_config'),
            'tokenizer_path': str(config.model_path), 'kv_cache_dtype': config.kv_cache_dtype,
            'requested_attention_backend': config.attention_backend,
            'experimental': config.experimental,
            'supports_vision': config.vision,
            'max_images_per_prompt': config.max_images_per_prompt if config.vision else None,
            'kv_scale_method': ('checkpoint scales when present, otherwise vLLM default 1.0; '
                                'actual values in workers.kv_cache_scales') if config.kv_cache_dtype != 'auto' else None,
            'supports_thinking': self.supports_thinking, 'supports_tools': self.supports_tools,
            'supports_reasoning_effort': self.supports_reasoning_effort,
            'reasoning_efforts': [e.value for e in ReasoningEffort] if self.supports_reasoning_effort else [],
            'workers': await self.llm.collective_rpc('runtime_status', timeout=60),
        }

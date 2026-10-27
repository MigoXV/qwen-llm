"""测试进程内注入 vLLM 参数；生产 V1Engine 不包含采集或实验分支。"""
from dataclasses import asdict
import os
from unittest.mock import patch

from qwen_llm.engine.vllm import V1Engine
from .config import ProfileConfig


class ExperimentEngine(V1Engine):
    def __init__(self, config, *, profiling: ProfileConfig):
        super().__init__(config)
        self.profiling = profiling

    async def start(self):
        from vllm.engine import arg_utils
        from vllm.config import ProfilerConfig
        original = arg_utils.AsyncEngineArgs
        settings = self.profiling
        experiment = self.config

        class DiagnosticArgs(original):
            def __init__(self, *args, **kwargs):
                kwargs['worker_cls'] = 'tests.performance.qwen38.worker.ProfilingNPUWorker'
                kwargs['profiler_config'] = ProfilerConfig(
                    profiler='torch', torch_profiler_dir=str(settings.output_dir), ignore_frontend=True)
                additional = dict(kwargs.get('additional_config') or {})
                additional['qwen_profile'] = asdict(settings)
                additional['qwen_profile'].pop('output_dir')
                additional['qwen_experiment'] = list(experiment.optimizations)
                additional['qwen_w4a8_linear_backend'] = experiment.w4a8_linear_backend.value
                kwargs['additional_config'] = additional
                if experiment.mixed_prefill_tokens is not None:
                    additional.setdefault('qwen38_adaptation', {})['mixed_prefill_tokens'] = experiment.mixed_prefill_tokens
                    kwargs['scheduler_cls'] = 'tests.performance.qwen38.mix_scheduler.Qwen38MixedScheduler'
                if 'cache' in experiment.optimizations:
                    kwargs['scheduler_cls'] = 'tests.performance.qwen38.cache.Qwen38AsyncScheduler'
                if 'checkpoint' in experiment.optimizations:
                    kwargs['scheduler_cls'] = 'tests.performance.qwen38.runtime_cache_candidate.Qwen38CheckpointScheduler'
                super().__init__(*args, **kwargs)

        with patch.dict(os.environ, {'VLLM_CUSTOM_SCOPES_FOR_PROFILING': '1'}), patch.object(
                arg_utils, 'AsyncEngineArgs', DiagnosticArgs):
            await super().start()
        if 'scale' in experiment.optimizations and any(
                x.get('experiment', {}).get('cached_scale_layers') != 128
                for x in self.runtime_identity.get('workers', [])):
            raise RuntimeError('未在全部128个FFN投影中启用尺度缓存')

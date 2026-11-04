"""测试进程内注入 vLLM 参数；生产 V1Engine 不包含采集或实验分支。"""
from dataclasses import asdict
import os
import hashlib
import json
from unittest.mock import patch

from qwen_llm.engine.vllm import V1Engine
from .config import ProfileConfig


def diagnostic_worker(config):
    if config.model_adaptation.value == 'qwen38-910b-v1':
        if config.optimizations or config.w4a8_linear_backend.value not in {'upstream','ascendc-w4a16','native-tuned'}:
            raise ValueError('生产基线不允许叠加旧实验适配或未验收后端')
        return 'tests.performance.qwen38.worker.ProfilingQwen38NPUWorker'
    return 'tests.performance.qwen38.worker.ProfilingNPUWorker'


class ExperimentEngine(V1Engine):
    def __init__(self, config, *, profiling: ProfileConfig):
        super().__init__(config)
        self.profiling = profiling

    async def start(self):
        if self.config.w4a8_linear_backend.value == 'ascendc-w4a16':
            from .ascendc_candidate import selection_rows
            selection_rows(json.loads(self.config.w4_selection.read_text()),
                           hashlib.sha256((self.config.model_path/'config.json').read_bytes()).hexdigest())
        from vllm.engine import arg_utils
        from vllm.config import ProfilerConfig
        original = arg_utils.AsyncEngineArgs
        settings = self.profiling
        experiment = self.config
        worker = diagnostic_worker(experiment)

        class DiagnosticArgs(original):
            def __init__(self, *args, **kwargs):
                kwargs['worker_cls'] = worker
                kwargs['profiler_config'] = ProfilerConfig(
                    profiler='torch', torch_profiler_dir=str(settings.output_dir), ignore_frontend=True)
                additional = dict(kwargs.get('additional_config') or {})
                additional['qwen_profile'] = asdict(settings)
                additional['qwen_profile'].pop('output_dir')
                additional['qwen_experiment'] = list(experiment.optimizations)
                additional['qwen_w4a8_linear_backend'] = experiment.w4a8_linear_backend.value
                if experiment.w4a8_linear_backend.value == 'ascendc-w4a16':
                    additional['qwen_w4_candidate'] = {
                        'selection':str(experiment.w4_selection.resolve()),
                        'host_library':str(experiment.w4_host_library.resolve()),
                        'selection_sha256':hashlib.sha256(experiment.w4_selection.read_bytes()).hexdigest(),
                        'host_sha256':hashlib.sha256(experiment.w4_host_library.read_bytes()).hexdigest()}
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

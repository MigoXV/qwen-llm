"""独立诊断 Worker：无逐步设备同步，主机时间不冒充设备执行时间。"""
from contextlib import nullcontext
from collections import Counter
from dataclasses import asdict
from functools import wraps
import time

import torch
import torch_npu

from qwen_llm.engine.ascend_worker import BudgetedNPUWorker


class ProfilingNPUWorker(BudgetedNPUWorker):
    def load_model(self):
        from .adaptations import install
        install(self.vllm_config)
        if (self.vllm_config.additional_config or {}).get('qwen_w4a8_linear_backend') == 'gmm':
            from .gmm import install_native_w4a8
            install_native_w4a8()
        super().load_model()
        self._case = None
        self._capture = False
        self._steps = []
        self._schedules = []
        self._spec_stats = []
        self._prompt_lengths = {}
        self._cache_pools = {}
        self._cache_evictions = Counter()
        self._cache_eviction_examples = []
        self._cache_min_free = {}
        self._cache_start_state = {}
        self._settings = self.vllm_config.additional_config["qwen_profile"]
        runner = self.model_runner
        determine = runner._determine_batch_execution_and_padding

        @wraps(determine)
        def execution_mode(*args, **kwargs):
            result = determine(*args, **kwargs)
            if self._case and self._steps:
                self._steps[-1].update(graph_mode=str(result[0]),
                                      padded_tokens=result[1].num_tokens)
            return result

        runner._determine_batch_execution_and_padding = execution_mode
        self._install_scheduler_observer()
        if self._settings["module_ranges"]:
            if not self.model_config.enforce_eager:
                raise ValueError("模块标记仅用于 Eager 映射，避免改变图捕获")
            self._install_module_ranges(runner.model)

    def runtime_status(self):
        result = super().runtime_status()
        modules = self.diagnostic_modules()
        head_weights = [m for m in modules if m['name'].endswith('lm_head')]
        scales = [m for m in modules if 'weight_scale_second' in m['parameters']]
        result['experiment'] = {
            'features': self.vllm_config.additional_config.get('qwen_experiment', []),
            'heads': head_weights,
            'cached_scale_layers': sum(m['parameters']['weight_scale_second']['dtype'] == 'torch.bfloat16' for m in scales),
            'gdn_output_int8_layers': sum(m['role'] == 'target' and m['name'].endswith('linear_attn.out_proj')
                and m['parameters'].get('weight', {}).get('dtype') == 'torch.int8' for m in modules)}
        return result

    def _install_scheduler_observer(self):
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.core.block_pool import BlockPool
        from vllm.v1.core.kv_cache_utils import get_group_id
        owner = self
        schedule = Scheduler.schedule
        stats = Scheduler.make_stats
        evict, allocate = BlockPool._maybe_evict_cached_block, BlockPool.get_new_blocks

        @wraps(evict)
        def observed_evict(pool, block):
            previous = block.block_hash
            result = evict(pool, block)
            if owner._case and result:
                group = get_group_id(previous)
                owner._cache_evictions[group] += 1
                if len(owner._cache_eviction_examples) < 20:
                    free = pool.free_block_queue.get_all_free_blocks()
                    owner._cache_eviction_examples.append({"group_id": group,
                        "at_ns": time.perf_counter_ns(), "remaining_free_blocks": len(free),
                        "remaining_uncached_free_blocks": sum(item.block_hash is None for item in free)})
            return result

        @wraps(allocate)
        def observed_allocate(pool, *args, **kwargs):
            result = allocate(pool, *args, **kwargs)
            owner._cache_pools[id(pool)] = pool
            if owner._case:
                owner._cache_min_free[id(pool)] = min(owner._cache_min_free.get(id(pool), pool.num_gpu_blocks),
                                                      pool.get_num_free_blocks())
            return result

        @wraps(schedule)
        def observed_schedule(scheduler, *args, **kwargs):
            start = time.perf_counter_ns()
            result = schedule(scheduler, *args, **kwargs)
            if owner._case:
                owner._schedules.append({"start_ns": start, "end_ns": time.perf_counter_ns(),
                    "tokens": result.total_num_scheduled_tokens,
                    "requests": list(result.num_scheduled_tokens)})
            return result

        @wraps(stats)
        def observed_stats(scheduler, *args, **kwargs):
            result = stats(scheduler, *args, **kwargs)
            spec = getattr(result, "spec_decoding_stats", None)
            if owner._case and spec is not None:
                owner._spec_stats.append(asdict(spec))
            return result

        Scheduler.schedule = observed_schedule
        Scheduler.make_stats = observed_stats
        BlockPool._maybe_evict_cached_block = observed_evict
        BlockPool.get_new_blocks = observed_allocate
        if set((self.vllm_config.additional_config or {}).get('qwen_experiment', [])) & {'cache', 'checkpoint'}:
            from .cache import FreeFirstBlockPool
            from .runtime_cache_candidate import Qwen38BlockPool
            selected_pool = Qwen38BlockPool if 'checkpoint' in self.vllm_config.additional_config.get('qwen_experiment', []) else FreeFirstBlockPool
            candidate_allocate = selected_pool.get_new_blocks
            @wraps(candidate_allocate)
            def observed_candidate(pool, *args, **kwargs):
                result = candidate_allocate(pool, *args, **kwargs)
                owner._cache_pools[id(pool)] = pool
                if owner._case:
                    owner._cache_min_free[id(pool)] = min(owner._cache_min_free.get(id(pool), pool.num_gpu_blocks), pool.get_num_free_blocks())
                return result
            selected_pool.get_new_blocks = observed_candidate

    def _install_module_ranges(self, model):
        # 单进程顺序 forward；只在 capture 时创建 record_function，图执行不装这些 hooks。
        for name, module in model.named_modules():
            if not name:
                continue
            contexts = []

            def before(layer, args, label=name, stack=contexts):
                context = torch.profiler.record_function("module/" + label) if self._capture else nullcontext()
                context.__enter__()
                stack.append(context)

            def after(layer, args, output, stack=contexts):
                stack.pop().__exit__(None, None, None)

            module.register_forward_pre_hook(before)
            module.register_forward_hook(after, always_call=True)

    def _create_profiler(self, trace_name):
        settings = self._settings
        return torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=settings["record_shapes"], with_stack=False,
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                aic_metrics=getattr(torch_npu.profiler.AiCMetrics, settings["aic_metrics"]),
                export_type=torch_npu.profiler.ExportType.Text,
                data_simplification=False),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                self.profiler_config.torch_profiler_dir, worker_name=trace_name))

    def diagnostic_begin_case(self, name: str, capture: bool = False):
        if self._case is not None:
            raise RuntimeError("已有 profiling case 未结束")
        self._steps, self._schedules, self._spec_stats = [], [], []
        self._device_events = []
        self._cache_evictions = Counter()
        self._cache_eviction_examples = []
        self._cache_min_free = {key: pool.get_num_free_blocks() for key, pool in self._cache_pools.items()}
        self._cache_start_state = {key: self._cache_state(pool) for key, pool in self._cache_pools.items()}
        if capture:
            self.profiler = None
        self._case, self._capture = name, capture
        return {"pid": __import__("os").getpid(), "case": name}

    @staticmethod
    def _cache_state(pool):
        from vllm.v1.core.kv_cache_utils import get_group_id
        cached, cached_free = Counter(), Counter()
        uncached_free = 0
        for block in pool.blocks:
            if block.is_null:
                continue
            if block.block_hash is not None:
                group = get_group_id(block.block_hash)
                cached[group] += 1
                if not block.ref_cnt:
                    cached_free[group] += 1
            elif not block.ref_cnt:
                uncached_free += 1
        index = getattr(pool, 'uncached_free', None)
        index_matches = index is None or set(index) == {b.block_id for b in pool.blocks
            if not b.is_null and b.ref_cnt == 0 and b.block_hash is None}
        metadata = {name: len(getattr(pool, name, {})) for name in
                    ('uncached_free', 'old_state_free', 'latest_checkpoint', 'checkpoint_owner', 'checkpoint_hash')}
        bounded = all(n <= pool.num_gpu_blocks for n in metadata.values())
        owners = getattr(pool, 'checkpoint_owner', {})
        latest = getattr(pool, 'latest_checkpoint', {})
        coherent = all(latest.get(key, (None,))[0] == block_id for block_id, key in owners.items())
        old_index = getattr(pool, 'old_state_free', None)
        old_matches = old_index is None or set(old_index) == {
            b.block_id for b in pool.blocks if not b.is_null and not b.ref_cnt
            and b.block_hash is not None and get_group_id(b.block_hash) in pool.gdn_groups
            and b.block_id not in owners}
        return {'allocator': type(pool).__name__, 'uncached_index_matches': index_matches, 'protected_checkpoints': len(getattr(pool, 'checkpoint_owner', {})), 'old_state_free': len(getattr(pool, 'old_state_free', {})),
                'total_blocks': pool.num_gpu_blocks, 'metadata_sizes': metadata,
                'metadata_bounded': bounded, 'checkpoint_owners_coherent': coherent,
                'old_state_index_matches': old_matches,
                "cached_blocks_by_group": dict(cached),
                "cached_free_blocks_by_group": dict(cached_free),
                "uncached_free_blocks": uncached_free}

    def diagnostic_allocator_state(self):
        return [self._cache_state(pool) for pool in self._cache_pools.values()]

    def diagnostic_end_case(self):
        if self._device_events:
            torch.npu.synchronize()
            origin = self._device_events[0][0]
            for step, events in zip(self._steps, self._device_events):
                start, forward_end, sample_end = events
                step["device_start_ms"] = origin.elapsed_time(start)
                step["device_forward_ms"] = start.elapsed_time(forward_end)
                if sample_end is not None:
                    step["device_sample_and_draft_ms"] = forward_end.elapsed_time(sample_end)
        result = {"case": self._case, "steps": self._steps,
                  "schedules": self._schedules, "spec_decoding": self._spec_stats,
                  "cache_evictions_by_group": dict(self._cache_evictions),
                  "cache_eviction_examples": self._cache_eviction_examples,
                  "cache_pools": [{"total_blocks": pool.num_gpu_blocks,
                      "allocatable_free_blocks_min": self._cache_min_free.get(key),
                      "allocatable_free_blocks_end": pool.get_num_free_blocks(),
                      "state_start": self._cache_start_state.get(key),
                      "state_end": self._cache_state(pool)}
                      for key, pool in self._cache_pools.items()]}
        self._case, self._capture = None, False
        return result

    def diagnostic_cache_layout(self):
        cache = self.model_runner.kv_cache_config
        states = []
        for name, module in self.model_runner.model.named_modules():
            if not hasattr(module, "in_proj_qkvz"):
                continue
            buffers = getattr(module, "kv_cache", ())
            if isinstance(buffers, (tuple, list)) and buffers and isinstance(buffers[0], (tuple, list)):
                states.append({"layer": name, "tensors": [
                    {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for value in buffers[0] if isinstance(value, torch.Tensor)]})
                break
        return {"total_blocks": cache.num_blocks, "sample_gdn_state": states, "groups": [
            {"group_id": index, "type": type(group.kv_cache_spec).__name__,
             "layers": len(group.layer_names), "block_size": group.kv_cache_spec.block_size,
             "page_size_bytes": group.kv_cache_spec.page_size_bytes,
             "state_shapes": getattr(group.kv_cache_spec, "shapes", None),
             "state_dtypes": [str(value) for value in (getattr(group.kv_cache_spec, "dtypes", ()) or ())],
             "mamba_cache_mode": getattr(group.kv_cache_spec, "mamba_cache_mode", None),
             "num_speculative_blocks": getattr(group.kv_cache_spec, "num_speculative_blocks", None)}
            for index, group in enumerate(cache.kv_cache_groups)]}

    def diagnostic_head_equivalence(self):
        with torch.inference_mode():
            model = self.model_runner.model
            target = model.get_language_model() if hasattr(model, 'get_language_model') else model
            draft = self.model_runner.drafter.model
            hidden = ((torch.arange(4*5120, device=self.device).reshape(4,5120) % 257 - 128).float() / 128).bfloat16()
            a, b = target.compute_logits(hidden), draft.compute_logits(hidden)
            return {'equal': torch.equal(a,b), 'max_abs': float((a-b).abs().max()),
                    'shared_module': target.lm_head is draft.lm_head,
                    'target_ptr': target.lm_head.weight.untyped_storage().data_ptr(),
                    'draft_ptr': draft.lm_head.weight.untyped_storage().data_ptr()}

    def diagnostic_modules(self):
        result = []
        models = [("target", self.model_runner.model)]
        draft = getattr(getattr(self.model_runner, "drafter", None), "model", None)
        if draft is not None:
            models.append(("draft", draft))
        for role, model in models:
            for name, module in model.named_modules():
                params = {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                                "storage_bytes": value.untyped_storage().nbytes(),
                                "storage_id": value.untyped_storage().data_ptr()}
                          for key, value in module.named_parameters(recurse=False)}
                if params:
                    result.append({"role": role, "name": name, "type": type(module).__name__, "parameters": params})
        return result

    def execute_model(self, scheduler_output):
        if not self._case or not scheduler_output.total_num_scheduled_tokens:
            return super().execute_model(scheduler_output)
        for request in scheduler_output.scheduled_new_reqs:
            self._prompt_lengths[request.req_id] = len(request.prompt_token_ids or [])
        computed = {r.req_id: r.num_computed_tokens for r in scheduler_output.scheduled_new_reqs}
        cached = scheduler_output.scheduled_cached_reqs
        computed.update(zip(cached.req_ids, cached.num_computed_tokens))
        scheduled = scheduler_output.num_scheduled_tokens
        prefills = [rid for rid in scheduled if computed.get(rid, 0) < self._prompt_lengths.get(rid, 0)]
        phase = "mixed" if prefills and len(prefills) < len(scheduled) else "prefill" if prefills else "decode"
        step = {"index": len(self._steps), "phase": phase, "start_ns": time.perf_counter_ns(),
                "scheduled_tokens": dict(scheduled), "prefill_requests": prefills,
                "spec_tokens": {k: len(v) for k, v in scheduler_output.scheduled_spec_decode_tokens.items()}}
        self._steps.append(step)
        if self._capture:
            events = [torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True), None]
            events[0].record()
            self._device_events.append(events)
        label = f"qwen_step/{step['index']}/{phase}"
        with torch.profiler.record_function(label) if self._capture else nullcontext():
            result = super().execute_model(scheduler_output)
        step["execute_return_ns"] = time.perf_counter_ns()
        if self._capture:
            self._device_events[-1][1].record()
        return result

    def sample_tokens(self, grammar_output):
        if not self._case:
            return super().sample_tokens(grammar_output)
        step = self._steps[-1]
        step["sample_start_ns"] = time.perf_counter_ns()
        with torch.profiler.record_function(f"qwen_sample/{step['index']}/{step['phase']}") if self._capture else nullcontext():
            result = super().sample_tokens(grammar_output)
        step["sample_return_ns"] = time.perf_counter_ns()
        if self._capture:
            self._device_events[-1][2] = torch.npu.Event(enable_timing=True)
            self._device_events[-1][2].record()
        return result

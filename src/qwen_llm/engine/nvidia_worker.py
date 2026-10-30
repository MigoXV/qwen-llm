"""只读 CUDA Worker 观测，不替换模型算子或缓存算法。"""
import os
from copy import copy
import torch
from vllm.v1.worker.gpu_worker import Worker


class ObservedGPUWorker(Worker):
    def runtime_status(self):
        from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config
        cache = self.model_runner.kv_cache_config
        # 只改浅拷贝中的 max_model_len，原引擎配置与实际分配池保持不动。
        view = copy(self.vllm_config)
        view.model_config = copy(self.vllm_config.model_config)
        maximum = self.vllm_config.model_config.max_model_len
        capacities = {}
        for length in sorted({*range(1024, maximum+1, 1024), maximum}):
            view.model_config.max_model_len = length
            capacities[str(length)] = get_max_concurrency_for_kv_cache_config(view, cache)
        cache_specs = []
        for group in cache.kv_cache_groups:
            root_spec = group.kv_cache_spec
            specs = getattr(root_spec, 'kv_cache_specs', {'group': root_spec}).values()
            kinds = {}
            for spec in specs:
                key = (type(spec).__name__, spec.block_size, str(getattr(spec, 'dtype', None)),
                       tuple(str(t) for t in getattr(spec, 'dtypes', ())))
                kinds[key] = {'kind': key[0], 'block_size': key[1], 'dtype': key[2],
                              'state_dtypes': list(key[3])}
            cache_specs.append({'layers': len(group.layer_names), 'page_bytes': root_spec.page_size_bytes,
                                'specs': list(kinds.values()), 'host_resident': group.host_resident})
        # 混合 Attention/GDN 缓存可能共用存储，按真实 storage 去重。
        storages = {}; dtypes = set()
        def visit(value):
            if isinstance(value, torch.Tensor):
                storage = value.untyped_storage()
                storages[storage.data_ptr()] = storage.nbytes()
                dtypes.add(str(value.dtype))
            elif isinstance(value, dict):
                for item in value.values(): visit(item)
            elif isinstance(value, (list, tuple)):
                for item in value: visit(item)
        visit(self.model_runner.kv_caches)
        scales = []; backends = set()
        for name, layer in self.vllm_config.compilation_config.static_forward_context.items():
            if hasattr(layer, 'attn_backend'):
                backends.add(layer.attn_backend.get_name())
            if hasattr(layer, '_k_scale'):
                scales.append({'layer': name, 'k': layer._k_scale.detach().cpu().tolist(),
                               'v': layer._v_scale.detach().cpu().tolist()})
        return {
            'pid': os.getpid(), 'device': str(self.device),
            'worker': f'{type(self).__module__}.{type(self).__name__}',
            'kv_cache_dtype': self.cache_config.cache_dtype,
            'prefix_caching': self.cache_config.enable_prefix_caching,
            'mamba_cache_mode': self.cache_config.mamba_cache_mode,
            'kv_cache_allocated_bytes': sum(storages.values()),
            'kv_cache_scales': scales,
            'cache_tensor_dtypes': sorted(dtypes), 'attention_backends': sorted(backends),
            'cache_specs': cache_specs,
            'full_context_capacity': get_max_concurrency_for_kv_cache_config(self.vllm_config, cache),
            'capacity_by_context': capacities,
            'allocated_bytes': torch.cuda.memory_allocated(), 'reserved_bytes': torch.cuda.memory_reserved(),
            'graph_mode': str(self.vllm_config.compilation_config.cudagraph_mode),
        }

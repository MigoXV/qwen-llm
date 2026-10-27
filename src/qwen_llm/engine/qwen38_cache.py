"""Qwen3.8-27B在1536对齐混合缓存中的提示检查点分配策略。"""
from collections import OrderedDict

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.sched.async_scheduler import AsyncScheduler


class Qwen38BlockPool(BlockPool):
    """先分配未缓存块，再回收早期GDN状态，最后遵守原LRU。

    检查点仅获得软保护，内存压力下仍可回收；不改变hash、引用计数、
    前缀匹配、MTP最后块裁剪或递归状态精度。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize_policy({1, 2, 3})

    def initialize_policy(self, gdn_groups):
        self.gdn_groups = set(gdn_groups)
        self.uncached_free = OrderedDict(
            (b.block_id, b) for b in self.free_block_queue.get_all_free_blocks()
            if b.block_hash is None and not b.is_null)
        self.old_state_free = OrderedDict()
        self.latest_checkpoint = {}
        self.checkpoint_owner = {}
        self.checkpoint_hash = {}

    def _old_state(self, block):
        from vllm.v1.core.kv_cache_utils import get_group_id
        if (not block.is_null and block.ref_cnt == 0 and block.block_hash is not None
                and get_group_id(block.block_hash) in self.gdn_groups
                and block.block_id not in self.checkpoint_owner):
            self.old_state_free[block.block_id] = block

    def _forget_checkpoint(self, block_id):
        key = self.checkpoint_owner.pop(block_id, None)
        if key is not None and self.latest_checkpoint.get(key, (None,))[0] == block_id:
            self.latest_checkpoint.pop(key, None)
        block = self.blocks[block_id]
        if block.block_hash is not None and self.checkpoint_hash.get(block.block_hash) == block_id:
            self.checkpoint_hash.pop(block.block_hash, None)

    def _unprotect(self, block_id):
        self._forget_checkpoint(block_id)
        self._old_state(self.blocks[block_id])

    def cache_full_blocks(self, request, blocks, num_cached_blocks, num_full_blocks,
                          block_size, kv_cache_group_id):
        super().cache_full_blocks(request, blocks, num_cached_blocks, num_full_blocks,
                                  block_size, kv_cache_group_id)
        if kv_cache_group_id not in self.gdn_groups:
            return
        # 与现有Eagle/MTP裁剪一致；较晚的状态不能用于同一提示的重放。
        safe_blocks = max(0, request.num_prompt_tokens // block_size - 1)
        key = (request.request_id, kv_cache_group_id)
        for index in range(num_cached_blocks, min(num_full_blocks, safe_blocks)):
            block = blocks[index]
            if block.is_null or block.block_hash is None:
                continue
            previous = self.latest_checkpoint.get(key)
            if previous is not None and previous[1] >= index:
                continue
            if previous is not None and previous[0] != block.block_id:
                self._unprotect(previous[0])
            duplicate = self.checkpoint_hash.get(block.block_hash)
            if duplicate is not None and duplicate != block.block_id:
                self._unprotect(duplicate)
            self.latest_checkpoint[key] = (block.block_id, index)
            self.checkpoint_owner[block.block_id] = key
            self.checkpoint_hash[block.block_hash] = block.block_id
            self.old_state_free.pop(block.block_id, None)

    def get_new_blocks(self, num_blocks):
        if num_blocks < 0 or num_blocks > self.get_num_free_blocks():
            raise ValueError('没有足够的空闲KV blocks')
        blocks = []
        for index in (self.uncached_free, self.old_state_free):
            for _ in range(min(num_blocks - len(blocks), len(index))):
                _, block = index.popitem(last=False)
                self.free_block_queue.remove(block)
                blocks.append(block)
        blocks.extend(self.free_block_queue.popleft_n(num_blocks - len(blocks)))
        for block in blocks:
            self.uncached_free.pop(block.block_id, None)
            self.old_state_free.pop(block.block_id, None)
            if self.enable_caching:
                self._maybe_evict_cached_block(block)
            assert block.ref_cnt == 0
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_allocated(block)
        return blocks

    def touch(self, blocks):
        for block in blocks:
            self.uncached_free.pop(block.block_id, None)
            self.old_state_free.pop(block.block_id, None)
        super().touch(blocks)

    def free_blocks(self, ordered_blocks):
        blocks = list(ordered_blocks)
        super().free_blocks(blocks)
        for block in blocks:
            if block.ref_cnt == 0 and not block.is_null and block.block_hash is None:
                self.uncached_free[block.block_id] = block
            self._old_state(block)

    def _maybe_evict_cached_block(self, block):
        previous = block.block_hash
        result = super()._maybe_evict_cached_block(block)
        if result:
            self.old_state_free.pop(block.block_id, None)
            if self.checkpoint_hash.get(previous) == block.block_id:
                self.checkpoint_hash.pop(previous, None)
            self._forget_checkpoint(block.block_id)
            if (block.ref_cnt == 0 and not block.is_null
                    and block.prev_free_block is not None and block.next_free_block is not None):
                self.uncached_free[block.block_id] = block
        return result

    def reset_prefix_cache(self):
        result = super().reset_prefix_cache()
        if result:
            self.initialize_policy(self.gdn_groups)
        return result


class Qwen38CheckpointScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import torch
        from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
        from .qwen38 import model_shape

        model_shape(self.vllm_config.model_config.model)
        if (self.vllm_config.parallel_config.tensor_parallel_size != 1
                or self.vllm_config.model_config.dtype != torch.bfloat16
                or self.num_spec_tokens != 2 or not self.use_eagle):
            raise ValueError('检查点策略只验证TP1、BF16和MTP2')
        manager = self.kv_cache_manager
        groups = manager.kv_cache_config.kv_cache_groups
        gdn_groups = {i for i, group in enumerate(groups) if isinstance(group.kv_cache_spec, MambaSpec)}
        if (len(groups) != 4 or gdn_groups != {1, 2, 3}
                or not isinstance(groups[0].kv_cache_spec, FullAttentionSpec)
                or len(groups[0].layer_names) != 17
                or any(group.kv_cache_spec.block_size != 1536 for group in groups)):
            raise ValueError('专项缓存要求1536对齐、17层全注意力及三组GDN状态')
        for index in gdn_groups:
            spec = groups[index].kv_cache_spec
            if (len(groups[index].layer_names) != 16 or spec.mamba_cache_mode != 'align'
                    or spec.num_speculative_blocks != 2
                    or tuple(tuple(s) for s in spec.shapes) != ((5, 10240), (48, 128, 128))
                    or tuple(spec.dtypes) != (torch.bfloat16, torch.float32)):
                raise ValueError('专项缓存要求已验证的FP32递归状态布局')
        old = manager.block_pool
        assert old.get_num_free_blocks() == old.num_gpu_blocks - 1
        pool = Qwen38BlockPool.__new__(Qwen38BlockPool)
        pool.__dict__.update(old.__dict__)
        pool.initialize_policy(gdn_groups)
        manager.block_pool = manager.coordinator.block_pool = pool
        for group in manager.coordinator.single_type_managers:
            assert group.block_pool is old
            group.block_pool = pool

"""保留原 LRU 队列，用独立 O(1) 索引优先分配未缓存空闲块。"""
from collections import OrderedDict
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.sched.async_scheduler import AsyncScheduler


class FreeFirstBlockPool(BlockPool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rebuild_uncached()

    def _rebuild_uncached(self):
        self.uncached_free = OrderedDict((b.block_id, b) for b in self.free_block_queue.get_all_free_blocks()
                                         if b.block_hash is None and not b.is_null)

    def get_new_blocks(self, num_blocks):
        if num_blocks > self.get_num_free_blocks() or num_blocks < 0:
            raise ValueError('没有足够的空闲 KV blocks')
        blocks = []
        for _ in range(min(num_blocks, len(self.uncached_free))):
            _, block = self.uncached_free.popitem(last=False)
            self.free_block_queue.remove(block)
            blocks.append(block)
        blocks.extend(self.free_block_queue.popleft_n(num_blocks - len(blocks)))
        for block in blocks:
            self.uncached_free.pop(block.block_id, None)
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
        super().touch(blocks)

    def free_blocks(self, ordered_blocks):
        blocks = list(ordered_blocks)
        super().free_blocks(blocks)
        for block in blocks:
            if block.ref_cnt == 0 and not block.is_null and block.block_hash is None:
                self.uncached_free[block.block_id] = block

    def cache_full_blocks(self, *args, **kwargs):
        super().cache_full_blocks(*args, **kwargs)
        # 新增 hash 的块均处于请求使用中，已不在未缓存空闲索引中。

    def _maybe_evict_cached_block(self, block):
        result = super()._maybe_evict_cached_block(block)
        if (result and block.ref_cnt == 0 and not block.is_null
                and block.prev_free_block is not None and block.next_free_block is not None):
            self.uncached_free[block.block_id] = block
        return result

    def reset_prefix_cache(self):
        result = super().reset_prefix_cache()
        if result:
            self._rebuild_uncached()
        return result


def attach(manager):
    """调度器初始化后尚无请求，替换全部 pool 引用，保留同一份 block 对象。"""
    old = manager.block_pool
    pool = FreeFirstBlockPool.__new__(FreeFirstBlockPool)
    pool.__dict__.update(old.__dict__)
    pool._rebuild_uncached()
    assert old.get_num_free_blocks() == old.num_gpu_blocks - 1
    manager.block_pool = manager.coordinator.block_pool = pool
    for group in manager.coordinator.single_type_managers:
        assert group.block_pool is old
        group.block_pool = pool
    return pool


class Qwen38AsyncScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from .adaptations import check_model
        check_model(self.vllm_config)
        attach(self.kv_cache_manager)


class CheckpointBlockPool(FreeFirstBlockPool):
    """优先回收较早的GDN状态，保护各前缀最后可重用的提示检查点。"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_checkpoints({1, 2, 3})

    def init_checkpoints(self, gdn_groups):
        self.gdn_groups = set(gdn_groups)
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
        # Hash may have been cleared by upstream eviction already.
        if block.block_hash is not None and self.checkpoint_hash.get(block.block_hash) == block_id:
            self.checkpoint_hash.pop(block.block_hash, None)

    def _unprotect(self, block_id):
        self._forget_checkpoint(block_id)
        self._old_state(self.blocks[block_id])

    def cache_full_blocks(self, request, blocks, num_cached_blocks, num_full_blocks, block_size, kv_cache_group_id):
        super().cache_full_blocks(request, blocks, num_cached_blocks, num_full_blocks, block_size, kv_cache_group_id)
        if kv_cache_group_id not in self.gdn_groups:
            return
        # 保持EAGLE最后一块裁剪语义，只保护可用于同一提示重放的状态。
        safe_blocks = max(0, request.num_prompt_tokens // block_size - 1)
        key = (request.request_id, kv_cache_group_id)
        for i in range(num_cached_blocks, min(num_full_blocks, safe_blocks)):
            block = blocks[i]
            if block.is_null or block.block_hash is None:
                continue
            previous = self.latest_checkpoint.get(key)
            if previous is not None and previous[1] >= i:
                continue
            if previous is not None and previous[0] != block.block_id:
                self._unprotect(previous[0])
            duplicate = self.checkpoint_hash.get(block.block_hash)
            if duplicate is not None and duplicate != block.block_id:
                self._unprotect(duplicate)
            self.latest_checkpoint[key] = (block.block_id, i)
            self.checkpoint_owner[block.block_id] = key
            self.checkpoint_hash[block.block_hash] = block.block_id
            self.old_state_free.pop(block.block_id, None)

    def get_new_blocks(self, num_blocks):
        if num_blocks > self.get_num_free_blocks() or num_blocks < 0:
            raise ValueError('没有足够的空闲 KV blocks')
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
            self.old_state_free.pop(block.block_id, None)
        super().touch(blocks)

    def free_blocks(self, ordered_blocks):
        blocks = list(ordered_blocks)
        super().free_blocks(blocks)
        for block in blocks:
            self._old_state(block)

    def _maybe_evict_cached_block(self, block):
        previous = block.block_hash
        result = super()._maybe_evict_cached_block(block)
        if result:
            self.old_state_free.pop(block.block_id, None)
            if self.checkpoint_hash.get(previous) == block.block_id:
                self.checkpoint_hash.pop(previous, None)
            self._forget_checkpoint(block.block_id)
        return result

    def reset_prefix_cache(self):
        result = super().reset_prefix_cache()
        if result:
            self.init_checkpoints(self.gdn_groups)
        return result


class Qwen38CheckpointScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from .adaptations import check_model
        from vllm.v1.kv_cache_interface import MambaSpec
        check_model(self.vllm_config)
        if self.num_spec_tokens != 2 or not self.use_eagle:
            raise ValueError('检查点保护只验证MTP2')
        manager = self.kv_cache_manager
        old = manager.block_pool
        pool = CheckpointBlockPool.__new__(CheckpointBlockPool)
        pool.__dict__.update(old.__dict__)
        pool._rebuild_uncached()
        groups = {i for i, group in enumerate(manager.kv_cache_config.kv_cache_groups)
                  if isinstance(group.kv_cache_spec, MambaSpec)}
        if len(groups) != 3 or any(manager.kv_cache_config.kv_cache_groups[i].kv_cache_spec.mamba_cache_mode != 'align' for i in groups):
            raise ValueError('检查点保护要求三组GDN状态')
        pool.init_checkpoints(groups)
        assert old.get_num_free_blocks() == old.num_gpu_blocks - 1
        manager.block_pool = manager.coordinator.block_pool = pool
        for group in manager.coordinator.single_type_managers:
            assert group.block_pool is old
            group.block_pool = pool

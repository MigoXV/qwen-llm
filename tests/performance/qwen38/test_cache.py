"""真实 vLLM BlockPool 的回归，覆盖引用、LRU 和分组 hash。"""
import random
import pytest

pytest.importorskip('vllm')
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash,make_block_hash_with_group_id
from .cache import FreeFirstBlockPool, CheckpointBlockPool
from qwen_llm.engine.qwen38_cache import Qwen38BlockPool


def cached(pool,blocks,group=1):
    for block in blocks:
        key=make_block_hash_with_group_id(BlockHash(bytes([block.block_id])*32),group)
        block.block_hash=key;pool.cached_block_hash_to_block.insert(key,block)


def invariant(pool):
    free=pool.free_block_queue.get_all_free_blocks()
    assert len(free)==pool.get_num_free_blocks()==len({b.block_id for b in free})
    assert all(b.ref_cnt==0 and not b.is_null for b in free)
    assert set(pool.uncached_free)=={b.block_id for b in free if b.block_hash is None}
    assert all(b.ref_cnt>=0 for b in pool.blocks)
    assert not any(b.is_null for b in free)
    if hasattr(pool, 'gdn_groups'):
        from vllm.v1.core.kv_cache_utils import get_group_id
        assert set(pool.old_state_free)=={b.block_id for b in free if b.block_hash is not None
            and get_group_id(b.block_hash) in pool.gdn_groups and b.block_id not in pool.checkpoint_owner}
        assert all(pool.latest_checkpoint[key][0]==block_id for block_id,key in pool.checkpoint_owner.items())
        for index in (pool.latest_checkpoint,pool.checkpoint_owner,pool.checkpoint_hash):
            assert len(index)<=pool.num_gpu_blocks


def test_preserves_cached_blocks_when_uncached_blocks_are_available():
    for cls,expected in [(BlockPool,0),(FreeFirstBlockPool,2)]:
        p=cls(8,True,1536)
        all_blocks=p.get_new_blocks(7)
        cached(p,all_blocks[:2])
        p.free_blocks(all_blocks)
        got=p.get_new_blocks(2)
        assert len(p.cached_block_hash_to_block)==expected
        assert all(b.ref_cnt==1 for b in got)
        if cls is FreeFirstBlockPool:invariant(p)


def test_falls_back_to_original_cached_lru_and_handles_reset_and_evict():
    p=FreeFirstBlockPool(5,True,1536)
    blocks=p.get_new_blocks(4);cached(p,blocks)
    assert not p.reset_prefix_cache()
    p.free_blocks(blocks)
    p.touch([blocks[1]]);p.free_blocks([blocks[1]])
    assert p.get_new_blocks(1)[0] is blocks[0]
    p.evict_blocks({blocks[2].block_id})
    assert p.get_new_blocks(1)[0] is blocks[2]
    p.free_blocks([blocks[0],blocks[2]])
    assert p.reset_prefix_cache()
    invariant(p)
    assert len(p.uncached_free)==4
    with pytest.raises(ValueError):p.get_new_blocks(5)


@pytest.mark.parametrize('pool_cls',[FreeFirstBlockPool,CheckpointBlockPool,Qwen38BlockPool])
def test_duplicate_hash_refcounts_and_randomized_allocation_lifecycle(pool_cls):
    p=pool_cls(31,True,1536)
    rng=random.Random(42);held=[]
    for _ in range(2000):
        if held and rng.random()<.5:
            b=held.pop(rng.randrange(len(held)))
            p.free_blocks([b])
        elif p.get_num_free_blocks():
            b=p.get_new_blocks(1)[0];held.append(b)
            if rng.random()<.5:
                cached(p,[b],rng.randrange(4))
            if rng.random()<.1:
                p.touch([b]);held.append(b)
        if rng.random()<.1:
            p.evict_blocks({rng.randrange(1,31)})
        invariant(p)
    p.free_blocks(held);assert p.reset_prefix_cache();invariant(p)
    b=p.get_new_blocks(2);key=make_block_hash_with_group_id(BlockHash(b'x'*32),1)
    for x in b:x.block_hash=key;p.cached_block_hash_to_block.insert(key,x)
    p.free_blocks(b);p.evict_blocks({b[0].block_id})
    assert p.cached_block_hash_to_block.get_one_block(key) is b[1]
    invariant(p)


@pytest.mark.parametrize('pool_cls',[CheckpointBlockPool,Qwen38BlockPool])
def test_checkpoint_policy_keeps_last_reusable_state_and_attention_chain(pool_cls):
    from types import SimpleNamespace
    from .cache import CheckpointBlockPool
    p=pool_cls(7,True,1536)
    blocks=p.get_new_blocks(6)
    cached(p,blocks[:4],group=0)
    hashes=[BlockHash(b'a'*32),BlockHash(b'b'*32)]
    request=SimpleNamespace(request_id='one',num_prompt_tokens=8192,block_hashes=hashes)
    p.cache_full_blocks(request,blocks[4:],0,2,1536,1)
    p.free_blocks(blocks)
    assert p.get_new_blocks(1)[0] is blocks[4]
    assert blocks[5].block_id in p.checkpoint_owner
    assert all(b.block_hash is not None for b in blocks[:4])
    # Identical prefix from a new request replaces the protection record,
    # while duplicate hashes remain managed by upstream BlockHashToBlockMap.
    req2=SimpleNamespace(request_id='two',num_prompt_tokens=8192,block_hashes=hashes)
    p.cache_full_blocks(req2,[p.null_block,blocks[4]],1,2,1536,1)
    p.free_blocks([blocks[4]])
    assert len(p.checkpoint_owner)==1
    assert p.get_new_blocks(1)[0] is blocks[5]
    p.free_blocks([blocks[5]])
    assert p.reset_prefix_cache()
    assert not p.checkpoint_owner and not p.old_state_free
    invariant(p)


@pytest.mark.parametrize('pool_cls',[CheckpointBlockPool,Qwen38BlockPool])
def test_checkpoint_protection_retains_mtp_pruned_prompt_boundary(pool_cls):
    from types import SimpleNamespace
    from .cache import CheckpointBlockPool
    p=pool_cls(3,True,1536)
    b=p.get_new_blocks(2)
    request=SimpleNamespace(request_id='one',num_prompt_tokens=4096,
        block_hashes=[BlockHash(b'a'*32),BlockHash(b'b'*32)])
    p.cache_full_blocks(request,b,0,2,1536,1)
    p.free_blocks(b)
    # floor(4096/1536)-1 = one reusable block; the newer state cannot
    # be used after the existing Eagle last-block pruning.
    assert p.get_new_blocks(1)[0] is b[1]
    assert b[0].block_id in p.checkpoint_owner


@pytest.mark.parametrize('pool_cls',[CheckpointBlockPool,Qwen38BlockPool])
def test_checkpoint_protection_is_soft_and_does_not_leak_when_every_block_is_protected(pool_cls):
    from types import SimpleNamespace
    p=pool_cls(5,True,1536)
    blocks=p.get_new_blocks(4)
    for i,b in enumerate(blocks):
        request=SimpleNamespace(request_id=str(i),num_prompt_tokens=4096,
            block_hashes=[BlockHash(bytes([i])*32)])
        p.cache_full_blocks(request,[b],0,1,1536,1)
    p.free_blocks(blocks)
    assert len(p.checkpoint_owner)==4
    got=p.get_new_blocks(4)
    assert got==blocks and not p.checkpoint_owner and not p.latest_checkpoint and not p.checkpoint_hash
    p.free_blocks(got)
    invariant(p)


@pytest.mark.parametrize('pool_cls',[FreeFirstBlockPool,CheckpointBlockPool,Qwen38BlockPool])
def test_candidate_preserves_upstream_allocation_metrics(pool_cls):
    from types import SimpleNamespace
    allocations=[]
    collector=SimpleNamespace(on_block_allocated=lambda b:allocations.append(b.block_id),
                              on_block_evicted=lambda b:None)
    p=pool_cls(5,True,1536,metrics_collector=collector)
    first=p.get_new_blocks(3);p.free_blocks(first)
    second=p.get_new_blocks(2)
    assert allocations==[b.block_id for b in first+second]

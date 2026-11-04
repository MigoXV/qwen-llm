from .summarize_kv_capacity import context_bounds,verified_long,graph_verified,residency_verified


def test_pool_bound_is_not_a_verified_long_request():
    pool={'capacity_by_context':{'65536':8,'131072':4,'262144':2}}
    assert context_bounds(pool)=={'1':262144,'2':262144,'4':131072}
    assert not verified_long({'valid':True,'pool':pool})


def test_long_capacity_requires_full_outputs_no_preemption_and_no_cache_hits():
    m={'valid':True,'long_context':131072,'long_full_lengths':True,'long_result':{
        'preemptions_delta':{'vllm:num_preemptions':0},'responses':[
            {'input_tokens':131008,'output_tokens':64,'cached_tokens':0} for _ in range(4)]}}
    assert verified_long(m)
    m['long_result']['preemptions_delta']['vllm:num_preemptions']=1
    assert not verified_long(m)
    m['long_result']['preemptions_delta']['vllm:num_preemptions']=0
    m['long_result']['responses'][0]['cached_tokens']=1024
    assert not verified_long(m)


def test_mtp_acceptance_requires_captured_draft_graph_and_no_eager():
    g={'mode':'FULL_DECODE_ONLY','enforce_eager':False,'graphs_captured':6,
       'draft_enforce_eager':False,'draft_graph_enabled':True}
    m={'valid':True,'mtp':2,'final_runtime':{'graph':g}}
    assert graph_verified(m)
    g['draft_enforce_eager']=True
    assert not graph_verified(m)
    g['draft_enforce_eager']=False;g['graphs_captured']=3
    assert not graph_verified(m)


def test_mtp_residency_requires_all_four_prompts_computed_together():
    m={'long_context':144384,'long_output_tokens':512,'final_runtime':{}}
    assert not residency_verified(m)
    sample={'request_ids':['a','b','c','d'],'prompt_tokens':[143872]*4,'computed_tokens':[143873]*4}
    m['final_runtime']['four_full_prompt_residency']=sample
    assert residency_verified(m)
    sample['computed_tokens'][-1]=1000
    assert not residency_verified(m)

from .analysis import custom_ffn_calls,module_group


def test_mixed_vector_partner_and_inverse_do_not_duplicate_matmul_coverage():
    events=[{'name':name,'args':{'Task Type':kind}} for name,kind in (
        ('stream_v2','MIX_AIC'),('stream_v2','MIX_AIV'),('stream_v2','MIX_AIV'),
        ('gemm_nz_v2','AI_CORE'),('restore_nz_v2','AI_VECTOR_CORE'),('reduce_parts','AI_VECTOR_CORE'))]
    value=custom_ffn_calls(events)
    assert value['calls']=={'matmul':2,'vector_partner':2,'inverse_layout':1,'reduce':1}
    assert value['unknown_matmul_roles']==0
    assert custom_ffn_calls([{'name':'stream_v2'}])['unknown_matmul_roles']==1
    assert module_group('restore_nz_v2')=='FFN V2 inverse layout'

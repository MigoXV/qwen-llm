"""C8 解码图捕获与逐步序列长度更新；不冻结预热时的KV长度。"""
from collections import Counter
import torch
import torch_npu

from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl,AscendC8AttentionBackendImpl
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.compilation.acl_graph import get_graph_params,update_graph_params_workspaces
from vllm_ascend.utils import weak_ref_tensors

captures=Counter();updates=Counter();sequence_max=0


def status():
    return {'captures':dict(captures),'updates':dict(updates),'maximum_updated_sequence_length':sequence_max}


def _kwargs(param,table,lengths):
    _,name,q,k,v,ks,ko,vs,vo,out,lse,block_size,heads,kv_heads,scale=param
    return dict(query=q,key=k,value=v,key_antiquant_scale=ks,key_antiquant_offset=ko,
        value_antiquant_scale=vs,value_antiquant_offset=vo,block_table=table,
        actual_seq_lengths_kv=lengths,num_heads=heads,num_key_value_heads=kv_heads,
        input_layout='BNSD',scale=scale,block_size=block_size,
        key_antiquant_mode=0,value_antiquant_mode=0,sparse_mode=0)


class GraphC8Attention(AscendC8AttentionBackendImpl):
    def _forward_c8_decode(self,query,metadata,output,layer):
        if not _EXTRA_CTX.capturing:return super()._forward_c8_decode(query,metadata,output,layer)
        self._prepare_c8_scales(layer,query.device)
        blocks,block_size,_,_=self.key_cache.shape
        batch=len(metadata.seq_lens_list);num_tokens=query.shape[0]
        q=query[:batch].unsqueeze(2)
        k=self.key_cache.view(blocks,block_size,-1);v=self.value_cache.view(blocks,block_size,-1)
        out=output[:batch].unsqueeze(2)
        lse=torch.empty(1,dtype=query.dtype,device=query.device)
        param=('c8',self._c8_layer_name,weak_ref_tensors(q),weak_ref_tensors(k),weak_ref_tensors(v),
            weak_ref_tensors(layer._c8_k_aq_scale),weak_ref_tensors(layer._c8_k_aq_offset),
            weak_ref_tensors(layer._c8_v_aq_scale),weak_ref_tensors(layer._c8_v_aq_offset),
            weak_ref_tensors(out),weak_ref_tensors(lse),block_size,self.num_heads,self.num_kv_heads,self.scale)
        kwargs=_kwargs(param,metadata.block_tables,metadata.seq_lens_list)
        params=get_graph_params();workspace=params.workspaces.get(num_tokens)
        if workspace is None:
            workspace=torch_npu._npu_fused_infer_attention_score_get_max_workspace(**kwargs)
            update_graph_params_workspaces(num_tokens,workspace)
        stream=torch.npu.current_stream();event=torch.npu.ExternalEvent()
        event.wait(stream);event.reset(stream)
        params.events[num_tokens].append(event);params.attn_params[num_tokens].append(param)
        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score.out(**kwargs,workspace=workspace,out=[out,lse])
        params.handles[num_tokens].append(torch.npu.graph_task_group_end(stream))
        captures[str(num_tokens)]+=1
        return output


def install_update():
    original=AscendAttentionBackendImpl.update_graph_params
    if getattr(original,'_c8_graph',False):return
    def update(stream,context,num_tokens,config,*args,**kwargs):
        global sequence_max
        params=get_graph_params();records=params.attn_params[num_tokens]
        if not records or not isinstance(records[0][0],str) or records[0][0]!='c8':
            return original(stream,context,num_tokens,config,*args,**kwargs)
        with torch.npu.stream(stream):
            for param,handle,event in zip(records,params.handles[num_tokens],params.events[num_tokens]):
                metadata=context.attn_metadata[param[1]]
                lengths=metadata.seq_lens_list
                sequence_max=max(sequence_max,max(lengths,default=0))
                torch.npu.graph_task_update_begin(stream,handle)
                torch_npu.npu_fused_infer_attention_score.out(**_kwargs(param,metadata.block_tables,lengths),
                    workspace=params.workspaces[num_tokens],out=[param[9],param[10]])
                torch.npu.graph_task_update_end(stream);event.record(stream)
                updates[str(num_tokens)]+=1
    update._c8_graph=True
    AscendAttentionBackendImpl.update_graph_params=staticmethod(update)


def install_layer(name,layer):
    install_update();layer.impl.__class__=GraphC8Attention;layer.impl._c8_layer_name=name

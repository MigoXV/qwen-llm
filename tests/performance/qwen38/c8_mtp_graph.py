"""MTP C8图：将每个查询展开为具有独立因果前缀长度的单查询。"""
from collections import Counter
from copy import copy
import math
import torch
import torch_npu
from vllm.forward_context import get_forward_context
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend,AscendAttentionBackendImpl,AscendC8AttentionBackendImpl,AscendAttentionState
from vllm_ascend.compilation.acl_graph import (get_graph_params,get_draft_graph_params,
    update_graph_params_workspaces,update_draft_graph_params_workspaces)
from vllm_ascend.utils import weak_ref_tensors

captures=Counter();updates=Counter();maximums=Counter()
sharing=False;shared_tables={};row_indices={};table_stats=Counter()


def configure_sharing(enabled):
    global sharing
    sharing=bool(enabled)


def causal_rows(cumulative,seq_lens,total):
    if len(cumulative)!=len(seq_lens):raise ValueError('查询与KV请求数不一致')
    owners=[];lengths=[];start=0
    for owner,(end,seq_len) in enumerate(zip(cumulative,seq_lens)):
        end=int(end);count=end-start
        if count<0:raise ValueError('查询偏移非递增')
        owners.extend([owner]*count)
        lengths.extend(max(1,int(seq_len)-count+j+1) for j in range(count));start=end
    if len(owners)>total:raise ValueError('查询超出捕获容量')
    owners.extend([0]*(total-len(owners)));lengths.extend([1]*(total-len(lengths)))
    return owners,lengths


def _params():return get_draft_graph_params() if _EXTRA_CTX.is_draft_model else get_graph_params()
def _role():return 'draft' if _EXTRA_CTX.is_draft_model else 'target'
def status():return {'captures':dict(captures),'updates':dict(updates),'maximum_updated_sequence_lengths':dict(maximums),
    'shared_tables':sharing,'table_operations':dict(table_stats)}


def refresh_table(table,source,owners,lengths,block_size):
    columns=min(source.shape[1],math.ceil(max(lengths)/block_size))
    if columns>table.shape[1]:raise ValueError('图块表容量不足')
    key=(str(table.device),tuple(owners))
    indices=row_indices.get(key) if sharing else None
    if indices is None:
        indices=torch.tensor(owners,dtype=torch.int32,device=table.device)
        table_stats['index_creations']+=1
        if sharing and len(row_indices)<512:row_indices[key]=indices
    else:table_stats['index_reuses']+=1
    table[:,:columns].copy_(torch.index_select(source[:,:columns],0,indices))
    table_stats['table_copies']+=1


def refresh_once(table,source,owners,lengths,block_size,seen):
    if not sharing:
        refresh_table(table,source,owners,lengths,block_size)
        return
    pointer=table.data_ptr();signature=(source.data_ptr(),tuple(owners),tuple(lengths),block_size)
    if pointer in seen:
        if seen[pointer]!=signature:raise ValueError('共享块表的层元数据不一致')
        table_stats['layer_reuses']+=1
        return
    refresh_table(table,source,owners,lengths,block_size)
    seen[pointer]=signature


def _kwargs(d,lengths):
    return dict(query=d['q'],key=d['k'],value=d['v'],key_antiquant_scale=d['ks'],key_antiquant_offset=d['ko'],
        value_antiquant_scale=d['vs'],value_antiquant_offset=d['vo'],block_table=d['table'],
        actual_seq_lengths_kv=lengths,num_heads=d['heads'],num_key_value_heads=d['kv_heads'],
        input_layout='BNSD',scale=d['scale'],block_size=d['block_size'],key_antiquant_mode=0,value_antiquant_mode=0,sparse_mode=0)


class MtpC8Attention(AscendC8AttentionBackendImpl):
    def _forward_c8_fused_infer_attention(self,query,key,value,metadata,output,layer):
        if metadata.attn_state==AscendAttentionState.SpecDecoding:return self._forward_c8_decode(query,metadata,output,layer)
        return super()._forward_c8_fused_infer_attention(query,key,value,metadata,output,layer)

    def _forward_c8_chunked_prefill(self,query,key,value,metadata,output,layer):
        if _EXTRA_CTX.capturing:
            if metadata.num_prefills:raise ValueError('本实验仅捕获decode图')
            return self._forward_c8_decode(query,metadata,output,layer)
        nd=metadata.num_decode_tokens;nr=metadata.num_decodes
        if nd<=nr:return super()._forward_c8_chunked_prefill(query,key,value,metadata,output,layer)
        owners,lengths=causal_rows(list(metadata.actual_seq_lengths_q[:nr]),metadata.seq_lens_list[:nr],nd)
        indices=torch.tensor(owners,dtype=torch.int32,device=query.device)
        table=torch.index_select(metadata.block_tables[:nr],0,indices)
        d=self._data(query[:nd],output[:nd],layer,table)
        result,_=torch_npu.npu_fused_infer_attention_score(**_kwargs(d,lengths));output[:nd].copy_(result.squeeze(2))
        if metadata.num_prefills:
            pf=copy(metadata);pf.num_decode_tokens=0;pf.num_decodes=0
            pf.actual_seq_lengths_q=[int(n)-nd for n in metadata.actual_seq_lengths_q[nr:]]
            pf.seq_lens_list=metadata.seq_lens_list[nr:];pf.block_tables=metadata.block_tables[nr:]
            super()._forward_c8_chunked_prefill(query[nd:],key[nd:] if key is not None else None,
                value[nd:] if value is not None else None,pf,output[nd:],layer)
        return output

    def _data(self,query,output,layer,table):
        self._prepare_c8_scales(layer,query.device)
        blocks,block_size,_,_=self.key_cache.shape
        return {'q':query.unsqueeze(2),'k':self.key_cache.view(blocks,block_size,-1),'v':self.value_cache.view(blocks,block_size,-1),
            'ks':layer._c8_k_aq_scale,'ko':layer._c8_k_aq_offset,'vs':layer._c8_v_aq_scale,'vo':layer._c8_v_aq_offset,
            'heads':self.num_heads,'kv_heads':self.num_kv_heads,'scale':self.scale,'block_size':block_size,
            'table':table,'out':output.unsqueeze(2),'lse':torch.empty(1,dtype=query.dtype,device=query.device),
            'query_total':query.shape[0]}

    def _forward_c8_decode(self,query,metadata,output,layer):
        total=query.shape[0];owners,lengths=causal_rows(list(metadata.actual_seq_lengths_q),metadata.seq_lens_list,total)
        params=_params();role=_role();stream=torch.npu.current_stream()
        if _EXTRA_CTX.capturing:
            context=get_forward_context();descriptor=context.batch_descriptor
            registry_total=descriptor.num_tokens if descriptor is not None else context.num_tokens
            event=torch.npu.ExternalEvent();event.wait(stream);event.reset(stream)
            params.events[registry_total].append(event)
            slot=self._table_use[total];self._table_use[total]+=1;table=self._tables[total][slot]
            if sharing:
                key=(role,registry_total,total,slot,metadata.block_tables.data_ptr())
                table=shared_tables.setdefault(key,table)
        else:
            indices=torch.tensor(owners,dtype=torch.int32,device=query.device)
            table=torch.index_select(metadata.block_tables,0,indices)
        data=self._data(query,output,layer,table);kwargs=_kwargs(data,lengths)
        if _EXTRA_CTX.capturing:
            workspace=params.workspaces.get(registry_total)
            if workspace is None:
                workspace=torch_npu._npu_fused_infer_attention_score_get_max_workspace(**kwargs)
                (update_draft_graph_params_workspaces if role=='draft' else update_graph_params_workspaces)(registry_total,workspace)
            saved={k:weak_ref_tensors(v) if isinstance(v,torch.Tensor) else v for k,v in data.items()}
            params.attn_params[registry_total].append(('c8mtp',self._c8_layer_name,saved))
            torch.npu.graph_task_group_begin(stream)
            torch_npu.npu_fused_infer_attention_score.out(**kwargs,workspace=workspace,out=[data['out'],data['lse']])
            params.handles[registry_total].append(torch.npu.graph_task_group_end(stream));captures[f'{role}:{registry_total}']+=1
        else:
            actual,_=torch_npu.npu_fused_infer_attention_score(**kwargs);output.copy_(actual.squeeze(2))
        return output


def install_layer(name,layer):
    original=AscendAttentionBackendImpl.update_graph_params
    if not getattr(original,'_mtp_c8',False):
        def update(stream,context,total,config,speculative_config=None,num_dcp_pcp_tokens=None,draft_attn_metadatas=None):
            params=_params();records=params.attn_params[total]
            if not records or not isinstance(records[0][0],str) or records[0][0]!='c8mtp':
                return original(stream,context,total,config,speculative_config,num_dcp_pcp_tokens,draft_attn_metadatas)
            role=_role();occurrences=Counter();seen={};metadata_rows={}
            with torch.npu.stream(stream):
                for record,handle,event in zip(records,params.handles[total],params.events[total]):
                    _,key,data=record
                    if role=='draft':metadata=draft_attn_metadatas[occurrences[key]][key];occurrences[key]+=1
                    else:metadata=context.attn_metadata[key]
                    row_key=(id(metadata),data['query_total'])
                    if not sharing or row_key not in metadata_rows:
                        metadata_rows[row_key]=causal_rows(list(metadata.actual_seq_lengths_q),metadata.seq_lens_list,data['query_total'])
                    owners,lengths=metadata_rows[row_key]
                    refresh_once(data['table'],metadata.block_tables,owners,lengths,data['block_size'],seen)
                    torch.npu.graph_task_update_begin(stream,handle)
                    torch_npu.npu_fused_infer_attention_score.out(**_kwargs(data,lengths),workspace=params.workspaces[total],out=[data['out'],data['lse']])
                    torch.npu.graph_task_update_end(stream);event.record(stream)
                    updates[f'{role}:{total}']+=1;maximums[role]=max(maximums[role],max(metadata.seq_lens_list,default=0))
        update._mtp_c8=True;AscendAttentionBackendImpl.update_graph_params=staticmethod(update)
    layer.impl.__class__=MtpC8Attention;layer.impl._c8_layer_name=name;layer.impl._table_use=Counter()
    config=layer.impl.vllm_config
    manager_block=config.cache_config.block_size
    kernel_block=AscendAttentionBackend.get_supported_kernel_block_sizes()[0]
    columns=math.ceil(config.model_config.max_model_len/manager_block)*(manager_block//kernel_block)+32
    layer.impl._tables={n:[torch.zeros((n,columns),dtype=torch.int32,device=layer.k_cache_scale.device) for _ in range(8)]
        for n in (1,2,3,4,6,12)}

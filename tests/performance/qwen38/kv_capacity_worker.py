"""隔离的 KV 容量实验；复用生产权重适配，不放宽生产配置。"""
from copy import copy
from importlib.metadata import version
import json
from pathlib import Path

import torch
from qwen_llm.engine.ascend_worker import BudgetedNPUWorker
from qwen_llm.engine.qwen38 import artifact, PACKAGES
from qwen_llm.engine.qwen38_worker import install_scale_cache, install_quantized_head, install_fast_decode,install_head_sharing


class KVCapacityWorker(BudgetedNPUWorker):
    def load_model(self):
        config=self.vllm_config
        vocab4=(config.additional_config or {}).get('vocab4',False)
        if vocab4:
            from .export_candidates import sha256
            path=Path(config.model_config.model)
            record=json.loads((path/'vocab4_manifest.json').read_text())
            if (path/'.quantization.incomplete').exists() or not record['export_complete']:
                raise ValueError('词表候选未完成导出')
            verified=artifact(record['baseline'],{n:version(n) for n in PACKAGES})
            if not all(sha256(path/r['path'])==r['sha256'] for r in record['files']):
                raise ValueError('词表候选分片哈希不匹配')
        else:verified=artifact(config.model_config.model,{n:version(n) for n in PACKAGES})
        self.mtp=config.speculative_config.num_speculative_tokens if config.speculative_config else 0
        if config.model_config.dtype!=torch.bfloat16 or self.mtp not in (0,2):
            raise ValueError('本隔离实验仅允许BF16残差及MTP0/2')
        if self.mtp:
            from .mtp_graph_registry import install
            install()
        install_scale_cache()
        if vocab4:
            from .vocab4 import install
            install()
        elif verified['recipe'] in {'head','both'}:install_quantized_head()
        if self.mtp:
            if vocab4:
                from .vocab4 import install_head_sharing as share_vocab4_head
                share_vocab4_head()
            else:install_head_sharing()
        super().load_model()
        self.fast_w4=install_fast_decode(self.model_runner.model)
        self.maxima={};self.collecting=False;self.attention_names=[]
        self.mode=config.additional_config['kv_capacity_mode']
        if self.mode=='int8' and self.mtp:
            from .c8_mtp_graph import configure_sharing
            configure_sharing(config.additional_config.get('share_c8_tables',False))
        scales=(json.loads(Path(config.additional_config['kv_scales']).read_text())
                if self.mode=='int8' else {})
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
        for name,layer in config.compilation_config.static_forward_context.items():
            if not isinstance(getattr(layer,'impl',None),AscendAttentionBackendImpl):continue
            self.attention_names.append(name)
            if self.mode=='int8':
                from vllm_ascend.quantization.methods.kv_c8 import AscendC8KVCacheAttentionMethod
                from vllm_ascend.quantization.method_adapters import AscendKVCacheMethod
                method=AscendC8KVCacheAttentionMethod({},name)
                with torch.device(self.device):method.create_weights(layer)
                for kind in ('k','v'):
                    getattr(layer,f'{kind}_cache_scale').data=torch.tensor(scales[name][kind],device=self.device)
                    getattr(layer,f'{kind}_cache_offset').data=torch.zeros(1024,device=self.device)
                method.process_weights_after_loading(layer)
                layer.quant_method=AscendKVCacheMethod(method)
                if self.mtp:from .c8_mtp_graph import install_layer
                else:from .c8_graph import install_layer
                install_layer(name,layer)
            else:
                original=layer.impl.forward
                def observe(attn_layer,query,key,value,kv_cache,attn_metadata,*args,_name=name,_original=original,**kwargs):
                    if self.collecting and attn_metadata is not None and key is not None:
                        n=attn_metadata.num_actual_tokens
                        row=self.maxima.setdefault(_name,{})
                        for kind,tensor in (('k',key),('v',value)):
                            maximum=tensor[:n].reshape(-1,1024).float().abs().amax(0)
                            row[kind]=torch.maximum(row[kind],maximum) if kind in row else maximum
                    return _original(attn_layer,query,key,value,kv_cache,attn_metadata,*args,**kwargs)
                layer.impl.forward=observe
        if len(self.attention_names)!=(17 if self.mtp else 16):raise RuntimeError(f'全注意力层数异常：{len(self.attention_names)}')
        self.residency_sample=None
        original_execute=self.model_runner.execute_model
        def execute(*args,**kwargs):
            result=original_execute(*args,**kwargs)
            batch=self.model_runner.input_batch;n=batch.num_reqs
            if n>=4:
                prompts=batch.num_prompt_tokens[:n].tolist()
                computed=batch.num_computed_tokens_cpu[:n].tolist()
                if min(prompts)>=32768 and all(c>=p for c,p in zip(computed,prompts)):
                    self.residency_sample={'request_ids':list(batch.req_ids[:n]),'prompt_tokens':prompts,'computed_tokens':computed}
            return result
        self.model_runner.execute_model=execute

    def begin_calibration(self):
        self.maxima={};self.collecting=True

    def export_scales(self):
        self.collecting=False
        if set(self.maxima)!=set(self.attention_names):raise RuntimeError('校准未覆盖所有全注意力层')
        return {n:{k:(v.clamp(min=1e-6)*1.2/127).cpu().tolist() for k,v in row.items()} for n,row in self.maxima.items()}

    def kv_status(self):
        from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config
        cache=self.model_runner.kv_cache_config
        view=copy(self.vllm_config);view.model_config=copy(view.model_config)
        capacity={}
        for length in range(1024,262145,1024):
            view.model_config.max_model_len=length
            capacity[str(length)]=get_max_concurrency_for_kv_cache_config(view,cache)
        groups=[]
        for g in cache.kv_cache_groups:
            s=g.kv_cache_spec
            groups.append({'layers':len(g.layer_names),'type':type(s).__name__,
                'block_size':s.block_size,'page_bytes':s.page_size_bytes,
                'dtype':str(getattr(s,'dtype',None)),'state_dtypes':[str(d) for d in getattr(s,'dtypes',())]})
        storages={};dtypes=set()
        def visit(x):
            if isinstance(x,torch.Tensor):
                st=x.untyped_storage();storages[st.data_ptr()]=st.nbytes();dtypes.add(str(x.dtype))
            elif isinstance(x,dict):
                for v in x.values():visit(v)
            elif isinstance(x,(tuple,list)):
                for v in x:visit(v)
        visit(self.model_runner.kv_caches)
        from vllm.compilation.counter import compilation_counter
        graph={'mode':str(self.vllm_config.compilation_config.cudagraph_mode),
            'graphs_captured':compilation_counter.num_cudagraph_captured,
            'enforce_eager':self.model_config.enforce_eager}
        if self.mode=='int8':
            if self.mtp:from .c8_mtp_graph import status
            else:from .c8_graph import status
            graph['c8']=status()
        if self.mtp:
            graph['draft_enforce_eager']=self.vllm_config.speculative_config.enforce_eager
            graph['draft_graph_enabled']=self.model_runner.drafter.use_cuda_graph
            from .mtp_graph_registry import aliases
            graph['draft_registry_aliases']=dict(aliases)
        return {'mode':self.mode,'mtp':self.mtp,'graph':graph,'four_full_prompt_residency':self.residency_sample,
            'fast_w4':self.fast_w4,'attention_layers':len(self.attention_names),
            'num_blocks':cache.num_blocks,'groups':groups,'capacity_by_context':capacity,
            'cache_storage_bytes':sum(storages.values()),'cache_tensor_dtypes':sorted(dtypes),
            'npu_allocated_bytes':torch.npu.memory_allocated(),'npu_reserved_bytes':torch.npu.memory_reserved()}

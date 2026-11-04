"""将合并draft图的各步参数登记到外层图键，避免后续步事件无人更新。"""
from collections import Counter
from types import SimpleNamespace

aliases=Counter()


class Alias:
    def __init__(self,data,key):self.data=data;self.key=key
    def __getitem__(self,key):
        if key!=self.key:
            if not aliases[f'{key}->{self.key}']:print(f'merged draft graph registry: {key} -> {self.key}',flush=True)
            aliases[f'{key}->{self.key}']+=1
        return self.data[self.key]
    def get(self,key,default=None):
        if key!=self.key:aliases[f'{key}->{self.key}']+=1
        return self.data.get(self.key,default)


def install():
    import vllm_ascend.attention.attention_v1 as attention
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX
    from vllm.forward_context import get_forward_context
    getter=attention.get_draft_graph_params
    if getattr(getter,'_merged_graph_alias',False):return
    updater=attention.update_draft_graph_params_workspaces
    def outer_key():
        context=get_forward_context();descriptor=context.batch_descriptor
        return descriptor.num_tokens if descriptor is not None else context.num_tokens
    def get():
        params=getter()
        if not (_EXTRA_CTX.is_draft_model and _EXTRA_CTX.capturing):return params
        key=outer_key()
        return SimpleNamespace(**{n:Alias(getattr(params,n),key)
            for n in ('events','attn_params','handles','workspaces')})
    def update(key,workspace):
        if _EXTRA_CTX.is_draft_model and _EXTRA_CTX.capturing:key=outer_key()
        updater(key,workspace)
    get._merged_graph_alias=True
    attention.get_draft_graph_params=get
    attention.update_draft_graph_params_workspaces=update
    # Installed MTP wraps one model invocation, then replays the same graph for
    # the next draft step before recording its attention events. Capture both
    # steps together, as the installed Eagle path already does. Every step
    # retains its own query/output buffers and external event.
    from vllm_ascend.spec_decode.eagle_proposer import SpecDecodeBaseProposer
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
    from vllm.config import CUDAGraphMode
    original=SpecDecodeBaseProposer._maybe_share_lm_head
    def merged(proposer,model):
        original(proposer,model)
        if proposer.method=='mtp' and proposer.use_cuda_graph:
            if not isinstance(proposer.model,ACLGraphWrapper):
                raise RuntimeError('安装栈的MTP图包装方式发生变化')
            proposer.model=proposer.model.unwrap()
            proposer._runnable=ACLGraphWrapper(proposer._run_merged_draft,
                proposer.vllm_config,runtime_mode=CUDAGraphMode.FULL,
                use_eagle=True,enable_enpu=proposer.enable_enpu)
            print('MTP graph captures both draft steps in one merged graph',flush=True)
    SpecDecodeBaseProposer._maybe_share_lm_head=merged

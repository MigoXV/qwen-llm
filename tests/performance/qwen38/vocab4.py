"""Ascend group128 INT4词表查表与输出头接线；仅供独立候选使用。"""
import hashlib
import torch
from torch.nn import Parameter

_embedding_op=None


def unpack_rows(packed,scales):
    low=packed.bitwise_and(15)
    high=packed.bitwise_right_shift(4).bitwise_and(15)
    q=torch.stack((low,high),dim=-1).flatten(-2)
    q=torch.where(q>=8,q-16,q)
    return (q.reshape(*q.shape[:-1],-1,128).to(scales.dtype)*scales.unsqueeze(-1)).flatten(-2)


def lookup_rows(packed,scales,lut):
    indices=(packed.to(torch.int32)+128).reshape(-1)
    q=torch.index_select(lut,0,indices).reshape(*packed.shape[:-1],-1)
    return (q.reshape(*q.shape[:-1],-1,128)*scales.unsqueeze(-1)).flatten(-2)


def install():
    global _embedding_op
    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod,ParallelLMHead
    from vllm.model_executor.utils import set_weight_attrs
    from vllm_ascend.ops.vocab_parallel_embedding import AscendVocabParallelEmbedding
    from vllm_ascend.quantization.method_adapters import AscendEmbeddingMethod
    from qwen_llm.engine.qwen38_worker import CachedScaleLinearMethod
    if _embedding_op is None:
        @torch.library.custom_op('qwen_llm_exp::vocab4_embedding',mutates_args=(),device_types='npu')
        def embedding_op(ids:torch.Tensor,weight:torch.Tensor,scales:torch.Tensor,lut:torch.Tensor)->torch.Tensor:
            flat=ids.reshape(-1)
            return lookup_rows(torch.index_select(weight,0,flat),torch.index_select(scales,0,flat),lut).reshape(*ids.shape,5120)
        @embedding_op.register_fake
        def fake(ids,weight,scales,lut):
            return torch.empty((*ids.shape,5120),dtype=scales.dtype,device=ids.device)
        _embedding_op=embedding_op

    class PackedEmbeddingMethod(UnquantizedEmbeddingMethod):
        def create_weights(self,layer,input_size_per_partition,output_partition_sizes,input_size,output_size,params_dtype,**attrs):
            rows=sum(output_partition_sizes)
            for name,shape,dtype in [('weight',(rows,input_size//2),torch.int8),
                                      ('weight_scale',(rows,input_size//128),torch.bfloat16)]:
                param=Parameter(torch.empty(shape,dtype=dtype),requires_grad=False)
                set_weight_attrs(param,{'output_dim':0,**attrs});layer.register_parameter(name,param)

        def process_weights_after_loading(self,layer):
            byte=torch.arange(-128,128,dtype=torch.int16)
            low=byte.bitwise_and(15);high=byte.bitwise_right_shift(4).bitwise_and(15)
            q=torch.stack((low,high),-1);q=torch.where(q>=8,q-16,q)
            layer.register_buffer('nibble_lut',q.to(device=layer.weight.device,dtype=torch.bfloat16))

        def embedding(self,layer,input_):
            return _embedding_op(input_,layer.weight,layer.weight_scale,layer.nibble_lut)

    class VocabConfig:
        def get_quant_method(self,layer,prefix):
            return (PackedHeadMethod(CachedScaleLinearMethod())
                    if isinstance(layer,ParallelLMHead) else PackedEmbeddingMethod())

    class PackedHeadMethod(AscendEmbeddingMethod):
        def create_weights(self,layer,*args,**kwargs):
            super().create_weights(layer,*args,**kwargs)
            # Ascend linear loaders call this packed_factor; the vocabulary
            # loader uses pack_factor for the same packed output dimension.
            layer.weight.pack_factor=layer.weight.packed_factor

        def process_weights_after_loading(self,layer):
            import torch_npu
            weight=layer.weight.data
            scales=layer.weight_scale_second.data
            first=layer.weight_scale.data.float()
            # Compare the loaded tensors before NZ/INT32 conversion. CANN
            # cannot run Equal/Identity on the packed INT32 NZ representation.
            digest=hashlib.sha256()
            for tensor in (weight,scales,first):
                digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
            layer._vocab4_fingerprint=digest.hexdigest()
            self.parts=[]
            # CANN W4 limits N to 65535. Split output channels; every dot
            # product still spans the original full K and group128 scales.
            for i,start in enumerate(range(0,248320,62080)):
                packed=weight[start//2:(start+62080)//2].transpose(0,1).contiguous()
                w=torch_npu.npu_format_cast(packed,29).view(torch.int32)
                s=(scales[start:start+62080].float()*first[start:start+62080]).transpose(0,1).contiguous().bfloat16()
                layer.register_buffer(f'vocab_weight_{i}',w)
                layer.register_buffer(f'vocab_scale_{i}',s)
                self.parts.append((w,s))
            layer.weight.data=self.parts[0][0]
            for name in ('weight_scale_second','weight_offset_second','weight_scale','weight_offset','scale_bias'):
                if hasattr(layer,name):delattr(layer,name)

        def apply(self,layer,x,bias=None):
            import torch_npu
            if bias is not None:raise ValueError('词表W4仅支持无偏置输出头')
            return torch.cat([torch_npu.npu_weight_quant_batchmatmul(x,w,antiquant_scale=s,
                antiquant_group_size=128) for w,s in self.parts],dim=-1)

    original=AscendVocabParallelEmbedding.__init__
    def initialize(self,num_embeddings,embedding_dim,params_dtype=None,org_num_embeddings=None,
                   padding_size=64,quant_config=None,prefix=''):
        if (num_embeddings,embedding_dim)!=(248320,5120):raise ValueError('词表INT4实验仅支持248320×5120')
        original(self,num_embeddings,embedding_dim,params_dtype,org_num_embeddings,padding_size,VocabConfig(),prefix)
    AscendVocabParallelEmbedding.__init__=initialize


def install_head_sharing():
    from vllm_ascend.spec_decode.eagle_proposer import SpecDecodeBaseProposer
    original=SpecDecodeBaseProposer._maybe_share_lm_head
    if getattr(original,'_vocab4_shared',False):return
    def share(proposer,model):
        original(proposer,model)
        target=model.get_language_model() if hasattr(model,'get_language_model') else model
        target_head=target.lm_head;draft_head=proposer.model.lm_head
        if (not getattr(target_head,'_vocab4_fingerprint',None)
                or target_head._vocab4_fingerprint!=getattr(draft_head,'_vocab4_fingerprint',None)
                or type(target_head.quant_method) is not type(draft_head.quant_method)):
            raise RuntimeError('目标与MTP词表W4输出头内容不一致，拒绝共享')
        proposer.model.lm_head=target_head;proposer._qwen38_head_shared=True
        print('vocab4 MTP head shared after loaded-tensor SHA256 equality',flush=True)
    share._vocab4_shared=True
    SpecDecodeBaseProposer._maybe_share_lm_head=share

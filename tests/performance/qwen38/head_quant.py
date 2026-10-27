"""为该模型显式传递输出头量化方法；上游Qwen构造器遗漏quant_config。"""
from functools import wraps
import json
from pathlib import Path


def install_quantized_head():
    from vllm.config import get_current_vllm_config
    from vllm_ascend.ops.vocab_parallel_embedding import AscendParallelLMHead
    from vllm_ascend.quantization.method_adapters import AscendEmbeddingMethod
    from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicLinearMethod
    original=AscendParallelLMHead.__init__
    if getattr(original,'_qwen38_w8_head',False):return

    class HeadConfig:
        def __init__(self, parent):
            self.parent=parent

        def get_quant_method(self, layer, prefix):
            return AscendEmbeddingMethod(AscendW8A8DynamicLinearMethod())

        def __getattr__(self,name):
            return getattr(self.parent,name)

    @wraps(original)
    def initialize(layer, num_embeddings, embedding_dim, *args, **kwargs):
        if args:
            raise ValueError('专项输出头适配要求显式关键字可选参数')
        if num_embeddings!=248320 or embedding_dim!=5120:
            raise ValueError('专项输出头适配只支持248320x5120词表')
        config=get_current_vllm_config()
        description=json.loads((Path(config.model_config.model)/'quant_model_description.json').read_text())
        if description.get('lm_head.weight')!='W8A8_DYNAMIC':
            raise ValueError('该候选不是独立导出的W8输出头')
        kwargs['quant_config']=HeadConfig(config.quant_config)
        original(layer,num_embeddings,embedding_dim,**kwargs)

    initialize._qwen38_w8_head=True
    AscendParallelLMHead.__init__=initialize

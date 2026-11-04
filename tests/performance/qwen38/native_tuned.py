"""旧实验入口复用主程序实现，避免维护第二套算子分派。"""


def install(worker):
    from qwen_llm.engine.qwen38_worker import install_fast_decode
    if str(worker.model_config.dtype) != 'torch.bfloat16':
        raise ValueError('native-tuned只验证BF16')
    existing=getattr(worker,'_qwen38_runtime',{}).get('w4_decode',{})
    worker._native_tuned_identity = (existing if existing.get('backend')=='native-tuned'
                                     else install_fast_decode(worker.model_runner.model))
    worker._qwen38_runtime['w4_decode'] = worker._native_tuned_identity

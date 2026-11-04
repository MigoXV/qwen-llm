import json
from .summarize_main_fast_w4 import summarize,RUNS,CASES


def fixtures(root):
    for name in RUNS:
        fast=name.startswith('B');value=106 if fast else 100 if name=='A1-native' else 102
        worker={'worker':'qwen_llm.engine.qwen38_worker.Qwen38NPUWorker','full_context_capacity':4.04,
            'speculative_tokens':2,'graph_mode':'FULL_DECODE_ONLY','graphs_captured':3,'npu_allocated_bytes':1000,
            'model_adaptation':{'w4_decode':{'backend':'native-tuned' if fast else 'upstream','layers':128,
                                           'capture_dispatch':{'6:1':128,'12:1':128} if fast else {}}}}
        m={'valid':True,'released':True,'port_released':True,'exit_code':0,'fast_w4_requested':fast,
            'implementation_sha256':{'source':'same'},'ready':{'runtime':{'workers':[worker]}},'cases':[]}
        for case in CASES:
            m['cases'].append({'name':case,'rounds':[{'valid':True,'inventory':{'isolated':True},
                'output_tokens':[512]*int(case[-1]),'tokens_per_second':value+i,'ttft_ms':[10]} for i in (-1,0,1)]})
        (root/name).mkdir();(root/name/'manifest.json').write_text(json.dumps(m))


def test_stable_gain_requires_every_independent_fast_start_to_exceed_native(tmp_path):
    # Keep process medians separate: a pooled average may conceal a slower run.
    fixtures(tmp_path);result=summarize(tmp_path)
    assert result['stable_positive'] and result['production_path_complete']
    p=tmp_path/'B2-fast/manifest.json';m=json.loads(p.read_text())
    for r in m['cases'][0]['rounds']:r['tokens_per_second']=100
    p.write_text(json.dumps(m));result=summarize(tmp_path)
    assert not result['stable_positive']
    assert not result['cases'][0]['stable_positive']


def test_measurement_code_or_worker_mismatch_invalidates_acceptance(tmp_path):
    fixtures(tmp_path)
    p=tmp_path/'B1-fast/manifest.json';m=json.loads(p.read_text());m['implementation_sha256']['source']='changed';p.write_text(json.dumps(m))
    result=summarize(tmp_path)
    assert not result['production_path_complete'] and not result['stable_positive']


def test_fast_backend_label_without_actual_capture_is_not_evidence(tmp_path):
    fixtures(tmp_path)
    p=tmp_path/'B1-fast/manifest.json';m=json.loads(p.read_text())
    m['ready']['runtime']['workers'][0]['model_adaptation']['w4_decode']['capture_dispatch']={}
    p.write_text(json.dumps(m))
    result=summarize(tmp_path)
    assert not result['production_path_complete'] and not result['stable_positive']


def test_handled_sigterm_requires_completed_lifespan_shutdown(tmp_path):
    fixtures(tmp_path)
    p=tmp_path/'B1-fast/manifest.json';m=json.loads(p.read_text());m['exit_code']=-15
    p.write_text(json.dumps(m))
    assert not summarize(tmp_path)['production_path_complete']
    (p.parent/'server.log').write_text('INFO: Application shutdown complete.\n')
    assert summarize(tmp_path)['production_path_complete']
    m['exit_code']=-9;p.write_text(json.dumps(m))
    assert not summarize(tmp_path)['production_path_complete']

import asyncio
import json

import pytest

from qwen_llm import benchmark


def test_contaminated_device_never_reports_valid_benchmark(monkeypatch,tmp_path):
    monkeypatch.setattr(benchmark,'inventory',lambda:{'isolated':False,'owners':[['1','ASR']]})
    path = tmp_path/'failure.json'
    with pytest.raises(RuntimeError,match='独占'):
        asyncio.run(benchmark.run_benchmark('http://localhost:8000/v1','qwen3-14b','hi',1,path))
    record = json.loads(path.read_text())
    assert not record['valid']
    assert not record['rounds']
    assert '独占' in record['error']


@pytest.mark.parametrize('extra,expected',[('',True),('| 4 0 | 2345 | ASR | 500 | NA |',False)])
def test_parse_real_npu_smi_process_columns(monkeypatch,extra,expected):
    text = '| 4 0 | 1234 | VLLMEngineCor | 30000 | NA |\n' + extra
    monkeypatch.setattr(benchmark.subprocess,'check_output',lambda command,**kwargs:text if command[0]=='npu-smi' else 'revision')
    result = benchmark.inventory()
    assert result['isolated'] is expected
    assert result['owners'][0]==('1234','VLLMEngineCor')

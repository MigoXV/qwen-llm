import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_llm import modelscope


def listing(content=b'weights',path='weights.safetensors'):
    return {'Success':True,'Data':{'Files':[{'Type':'blob','Path':path,
        'Size':len(content),'Sha256':hashlib.sha256(content).hexdigest(),'Revision':'revision'}]}}


def mock_listing(monkeypatch,data):
    monkeypatch.setattr(modelscope.httpx,'get',lambda *a,**k:SimpleNamespace(
        raise_for_status=lambda:None,json=lambda:data))


def test_verified_download_and_resume(monkeypatch,tmp_path):
    mock_listing(monkeypatch,listing())
    calls=[]
    def download(args,**kwargs):
        calls.append(args)
        Path(args[args.index('--output')+1]).write_bytes(b'weights')
    monkeypatch.setattr(modelscope.subprocess,'run',download)
    modelscope.download_modelscope('owner/model',tmp_path,1)
    modelscope.download_modelscope('owner/model',tmp_path,1)
    assert len(calls)==1
    assert json.loads((tmp_path/'.modelscope-manifest.json').read_text())['verified']


def test_corrupt_download_is_not_promoted(monkeypatch,tmp_path):
    mock_listing(monkeypatch,listing())
    def download(args,**kwargs):
        Path(args[args.index('--output')+1]).write_bytes(b'corrupt')
    monkeypatch.setattr(modelscope.subprocess,'run',download)
    with pytest.raises(RuntimeError,match='校验失败'):
        modelscope.download_modelscope('owner/model',tmp_path,1)
    assert not (tmp_path/'weights.safetensors').exists()
    assert not json.loads((tmp_path/'.modelscope-manifest.json').read_text())['verified']


def test_reject_traversal(monkeypatch,tmp_path):
    mock_listing(monkeypatch,listing(path='../outside'))
    with pytest.raises(ValueError,match='不安全'):
        modelscope.download_modelscope('owner/model',tmp_path,1)

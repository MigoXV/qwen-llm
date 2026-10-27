import json
from pathlib import Path
import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import save_file,load_file
from .export_candidates import export,sha256


def test_independent_candidate_preserves_parent_and_untouched_tensors(tmp_path):
    source=tmp_path/'original';base=tmp_path/'baseline';out=tmp_path/'candidate'
    source.mkdir();base.mkdir()
    cfg={'model_type':'qwen3_5','text_config':{'num_hidden_layers':64}}
    for root in (source,base):(root/'config.json').write_text(json.dumps(cfg))
    weight=torch.arange(128).reshape(8,16).bfloat16()/13
    save_file({'lm_head.weight':weight},source/'original.safetensors')
    (source/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'lm_head.weight':'original.safetensors'}}))
    extra=torch.tensor([[1,2,3]],dtype=torch.int8)
    save_file({'lm_head.weight':weight,'untouched.weight':extra},base/'one.safetensors')
    (base/'quant_model_weights.safetensors.index.json').write_text(json.dumps({'weight_map':{'lm_head.weight':'one.safetensors','untouched.weight':'one.safetensors'}}))
    (base/'quant_model_description.json').write_text(json.dumps({'lm_head.weight':'FLOAT','untouched.weight':'W4A8_DYNAMIC'}))
    before=sha256(base/'one.safetensors')
    m=export(source,base,out,'head')
    assert m['export_complete'] and not m['validated_for_inference']
    assert not (out/'.quantization.incomplete').exists()
    assert sha256(base/'one.safetensors')==before
    exported=load_file(out/'one.safetensors')
    assert torch.equal(exported['untouched.weight'],extra)
    assert exported['lm_head.weight'].dtype==torch.int8
    assert (base/'one.safetensors').stat().st_ino!=(out/'one.safetensors').stat().st_ino
    with pytest.raises(ValueError,match='存在'):export(source,base,out,'head')
    with pytest.raises(ValueError,match='完全分开'):export(source,base,source/'new','head')

"""核对真实量化产物的分片、INT4 打包、MTP 与分词文件，不加载到 GPU。"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct

import typer

from tests.performance.nvidia.quantize import sha256


def tensors(directory):
    index = json.loads((directory/'model.safetensors.index.json').read_text())
    result = {}
    for filename in sorted(set(index['weight_map'].values())):
        with (directory/filename).open('rb') as file:
            size = struct.unpack('<Q', file.read(8))[0]
            header = json.loads(file.read(size))
        for name, spec in header.items():
            if name == '__metadata__':
                continue
            result[name] = {**spec, 'path': directory/filename, 'data_start': 8+size}
    if set(result) != set(index['weight_map']):
        raise ValueError('索引与实际分片的张量集合不一致')
    for name, filename in index['weight_map'].items():
        if result[name]['path'].name != filename:
            raise ValueError(f'索引指向错误分片：{name}')
    return index, result


def tensor_hash(spec):
    start, end = spec['data_offsets']
    digest = hashlib.sha256()
    with spec['path'].open('rb') as file:
        file.seek(spec['data_start']+start)
        remaining = end-start
        while remaining:
            data = file.read(min(8*1024**2, remaining))
            if not data:
                raise ValueError('权重文件提前结束')
            digest.update(data)
            remaining -= len(data)
    return digest.hexdigest()


def main(model: Path, output: Path = typer.Option(...), snapshot: Path | None = typer.Option(None)):
    manifest = json.loads((model/'quantization_manifest.json').read_text())
    if not manifest['export_complete'] or (model/'.quantization.incomplete').exists():
        raise ValueError('量化尚未完成')
    source = Path(manifest['source'])
    _, original = tensors(source)
    index, exported = tensors(model)
    source_config = json.loads((source/'config.json').read_text())
    exported_config = json.loads((model/'config.json').read_text())
    changed_config_keys = sorted(k for k in source_config.keys() | exported_config.keys()
                                 if source_config.get(k) != exported_config.get(k))
    config_preserved = set(changed_config_keys) <= {'quantization_config'}
    quantization = exported_config.get('quantization_config', {})
    groups = list(quantization.get('config_groups', {}).values())
    int4_group128 = bool(groups) and all(
        group.get('weights', {}).get('num_bits') == 4
        and group['weights'].get('group_size') == 128
        and group['weights'].get('symmetric') is True for group in groups)
    packed = {name: spec for name, spec in exported.items() if name.endswith('.weight_packed')}
    replacements = {name.removesuffix('_packed') for name in packed}
    missing = sorted(set(original)-set(exported)-replacements)
    mtp = []
    for name, old in original.items():
        if 'mtp' not in name:
            continue
        new = exported.get(name)
        same = bool(new and old['dtype'] == new['dtype'] and old['shape'] == new['shape']
                    and tensor_hash(old) == tensor_hash(new))
        mtp.append({'name':name,'unchanged':same})
    tokenizer = {}
    for name in ('tokenizer.json','tokenizer_config.json','chat_template.jinja','generation_config.json',
                 'vocab.json','merges.txt','ABLIT_META.json','LICENSE'):
        if (source/name).exists():
            tokenizer[name] = (model/name).is_file() and sha256(source/name) == sha256(model/name)
    shards = [{**item,'hash_matches':sha256(model/item['path']) == item['sha256']}
              for item in manifest['files']]
    source_check = None
    if snapshot is not None:
        frozen = json.loads(snapshot.read_text())
        if Path(frozen['source']).resolve() != source.resolve():
            raise ValueError('源模型快照目录不匹配')
        source_check = [{'path':item['path'],'unchanged':sha256(source/item['path']) == item['sha256']}
                        for item in frozen['files']]
    tensor_bytes = sum(x['data_offsets'][1]-x['data_offsets'][0] for x in exported.values())
    record = {
        'model':str(model.resolve()), 'source':str(source.resolve()),
        'method':manifest['method'], 'source_tensors':len(original), 'exported_tensors':len(exported),
        'model_config_preserved':config_preserved, 'config_changed_keys':changed_config_keys,
        'packed_linear_modules':len(packed), 'packed_dtypes':dict(Counter(x['dtype'] for x in packed.values())),
        'all_tensor_dtypes':dict(Counter(x['dtype'] for x in exported.values())),
        'scheme':manifest['scheme'], 'exported_quantization':quantization,
        'int4_group128':int4_group128, 'missing_source_tensors':missing,
        'dtype_changes':{n:[original[n]['dtype'],s['dtype']] for n,s in exported.items()
                        if n in original and s['dtype'] != original[n]['dtype']},
        'mtp':mtp, 'mtp_count_matches_manifest':manifest.get('preserved_mtp_tensors') == len(mtp),
        'tokenizer_and_metadata':tokenizer, 'shards':shards,
        'shard_bytes':sum(x['bytes'] for x in shards), 'tensor_bytes':tensor_bytes,
        'index_total_size':index.get('metadata',{}).get('total_size'),
        'source_snapshot_check':source_check,
    }
    record['passed'] = config_preserved and int4_group128 and bool(packed) and all(s['dtype'] == 'I32' for s in packed.values()) and not missing \
        and record['mtp_count_matches_manifest'] and all(x['unchanged'] for x in mtp) and all(tokenizer.values()) \
        and all(x['hash_matches'] for x in shards) \
        and (source_check is None or all(x['unchanged'] for x in source_check))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(record,ensure_ascii=False,indent=2))
    typer.echo(json.dumps({k:record[k] for k in ('method','packed_linear_modules','packed_dtypes',
        'missing_source_tensors','shard_bytes','tensor_bytes','index_total_size','passed')},indent=2))
    if not record['passed']:
        raise typer.Exit(1)


if __name__ == '__main__':
    typer.run(main)

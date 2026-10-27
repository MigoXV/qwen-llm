"""只读校验原始和已部署模型的完整分片哈希，结果写入报告目录。"""
import argparse
import json
from pathlib import Path
from .export_candidates import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();results=[]
    specs=[(Path('/home/model-bin/Qwen/Qwen3.8-27B'),Path('/home/qwen-quantize-20261016/reports/source-manifest.json')),
           (Path('/home/model-bin/Qwen/Qwen3.8-27B-W4A8-Ascend'),Path('/home/model-bin/Qwen/Qwen3.8-27B-W4A8-Ascend/quantization_manifest.json'))]
    for root,manifest in specs:
        files=json.loads(manifest.read_text())['files']
        for row in files:
            file=root/row['path']
            if not file.is_file():raise FileNotFoundError(file)
            actual=sha256(file);expected=row['sha256']
            result={'root':str(root),'path':row['path'],'sha256':actual,'expected':expected,'equal':actual==expected}
            results.append(result);print(json.dumps(result),flush=True)
    okay=all(x['equal'] for x in results)
    a.output.write_text(json.dumps({'passed':okay,'files':results},indent=2))
    if not okay:raise SystemExit(1)


if __name__=='__main__':main()

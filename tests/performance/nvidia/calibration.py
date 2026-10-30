"""固定 revision 的512条校准数据；两种候选复用同一 tokenized JSONL。"""
import json
from pathlib import Path
import hashlib
import gc
import typer

app=typer.Typer()

@app.command()
def main(model:Path=typer.Option(...),output:Path=typer.Option(...),samples:int=typer.Option(512)):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    if output.exists():raise ValueError('校准文件已存在')
    output.parent.mkdir(parents=True, exist_ok=True)
    repo='mlabonne/open-perfectblend'
    revision=HfApi().dataset_info(repo).sha
    ds=load_dataset(repo,split='train',streaming=True,revision=revision)
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    rows=[]
    iterator=iter(ds)
    try:
        for row in iterator:
            messages=[{'role':{'human':'user','gpt':'assistant'}.get(m['from'],m['from']),'content':m['value']} for m in row['conversations']]
            try:
                ids=tok.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=False,enable_thinking=False)[:4096]
            except (ValueError, TypeError, KeyError):
                continue
            if len(ids)<128:continue
            rows.append({'input_ids':ids,'attention_mask':[1]*len(ids)})
            if len(rows)%32==0:print('samples',len(rows),flush=True)
            if len(rows)==samples:break
    finally:
        iterator.close()
        del iterator, ds
        gc.collect()
    if len(rows)!=samples:raise RuntimeError('校准样本不足')
    output.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    output.with_suffix('.manifest.json').write_text(json.dumps({'dataset':repo,'revision':revision,'selection':f'first {samples} valid samples >=128 tokens; max4096; original order',
        'samples':samples,'token_lengths':[len(x['input_ids']) for x in rows],'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'seed':42},indent=2))

if __name__=='__main__':app()

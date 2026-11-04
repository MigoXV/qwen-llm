"""在NPU图中验证C8多查询、变长与右下因果遮罩。"""
import argparse
import json
from pathlib import Path
import torch
import torch_npu
from .benchmark import inventory


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or inventory()['owners']:raise RuntimeError('需要新目录与空闲NPU')
    a.output.mkdir(parents=True);torch.npu.set_device(0);torch.manual_seed(42)
    q=torch.randn(2,24,3,256,device='npu',dtype=torch.bfloat16)
    k=torch.randn(4,128,4,256,device='npu',dtype=torch.bfloat16);v=torch.randn_like(k)
    ks=(k.float().abs().amax((0,1))/127).clamp(min=1e-6);vs=(v.float().abs().amax((0,1))/127).clamp(min=1e-6)
    ki=(k.float()/ks).round().clamp(-128,127).to(torch.int8);vi=(v.float()/vs).round().clamp(-128,127).to(torch.int8)
    table=torch.tensor([[0],[1]],device='npu',dtype=torch.int32)
    mask=torch.triu(torch.ones(2048,2048,dtype=torch.bool),diagonal=1).to('npu')
    kw=dict(query=q,block_table=table,block_size=128,input_layout='BNSD',num_heads=24,
        num_key_value_heads=4,actual_seq_lengths=[3,2],actual_seq_lengths_kv=[128,127],
        atten_mask=mask,sparse_mode=3,scale=.0625)
    result={'graph':True,'passed':False,'query_lengths':[3,2]}
    owners=torch.tensor([0,0,0,1,1],device='npu',dtype=torch.int32)
    expanded=torch.index_select(table,0,owners)
    query=torch.cat((q[0,:,:3].transpose(0,1),q[1,:,:2].transpose(0,1))).unsqueeze(2).contiguous()
    try:
        graph=torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            expected,_=torch_npu.npu_fused_infer_attention_score(key=k.flatten(2),value=v.flatten(2),**kw)
            actual,_=torch_npu.npu_fused_infer_attention_score(query=query,key=ki.flatten(2),value=vi.flatten(2),
                key_antiquant_scale=ks.bfloat16().view(1,4,1,256),value_antiquant_scale=vs.bfloat16().view(1,4,1,256),
                key_antiquant_mode=0,value_antiquant_mode=0,block_table=expanded,block_size=128,
                input_layout='BNSD',num_heads=24,num_key_value_heads=4,
                actual_seq_lengths_kv=[126,127,128,126,127],sparse_mode=0,scale=.0625)
        graph.replay();torch.npu.synchronize()
        e=torch.cat((expected[0,:,:3].transpose(0,1),expected[1,:,:2].transpose(0,1))).float()
        y=actual.squeeze(2).float()
        error=((y-e).norm()/e.norm()).item();result.update(relative_l2=error,passed=error<=.05)
    except Exception as error:result['error']=str(error)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if not result['passed']:raise RuntimeError('MTP C8因果图算子未通过')


if __name__=='__main__':main()

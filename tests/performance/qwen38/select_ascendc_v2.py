"""两次独立完整复核取交集，不以某一层最快配置冒充可接入配置。"""
import argparse
import json
from pathlib import Path
from .ascendc_v2_candidate import validate_selection
from .ascendc_v2 import Tiling


def select(first,second,layout):
    for name in ('environment','raw_library_sha256','host_sha256','model_config_sha256','python_source_sha256','timing_schema'):
        if first.get(name)!=second.get(name):raise ValueError('两次复核环境或代码不一致：'+name)
    if first.get('run_id')==second.get('run_id'):raise ValueError('不能重复使用同一复核')
    def key(p):return (p['M'],p['N'],p['K'],p['layout'],*p['configuration'])
    a={key(p):p for p in first['qualified_profiles'] if p['layout']==layout}
    b={key(p):p for p in second['qualified_profiles'] if p['layout']==layout}
    byshape={}
    for k in sorted(a.keys()&b.keys()):
        item=dict(a[k]);item['worst_two_run_time_ratio']=max(a[k]['worst_layer_time_ratio'],b[k]['worst_layer_time_ratio'])
        byshape.setdefault(k[:3],[]).append(item)
    profiles=[]
    for shape,values in byshape.items():
        choice=min(values,key=lambda p:(p['worst_two_run_time_ratio'],Tiling(*p['configuration']).workspace_bytes(*shape),p['configuration']))
        profiles.append(choice)
    result={k:first[k] for k in ('schema_version','layout_version','environment','raw_library_sha256','host_sha256','model_config_sha256','python_source_sha256')}
    result.update(layout=layout,passed_micro_gate=bool(profiles),profiles=profiles,
                  validation_runs=[{**run,'samples':[r for r in run['samples'] if r['layout']==layout]} for run in (first,second)])
    if profiles:validate_selection(result,result['model_config_sha256'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--first',type=Path,required=True);p.add_argument('--second',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('输出必须不存在')
    a,b=[json.loads(path.read_text()) for path in (args.first,args.second)]
    # A negative decision also needs complete evidence. Never turn a truncated
    # validation file into "no eligible profiles" by inspecting only its list.
    from .summarize_ascendc_v2 import micro_summary
    micro_summary(a,b)
    args.output.mkdir(parents=True);decisions=[]
    for layout in (0,1):
        record=select(a,b,layout);path=args.output/f'layout{layout}.json';path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
        decisions.append({'layout':layout,'passed_micro_gate':record['passed_micro_gate'],'profiles':len(record['profiles']),'selection':str(path.resolve())})
    (args.output/'decision.json').write_text(json.dumps(decisions,indent=2)+'\n');print(json.dumps(decisions))


if __name__=='__main__':main()

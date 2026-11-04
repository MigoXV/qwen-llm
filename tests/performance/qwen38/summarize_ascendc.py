"""从当前基线与独立算子原始结果生成报告，不将单算子速度当成服务TPS。"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from .acceptance import stability_summary


def summarize(root):
    base = json.loads((root/'baseline/manifest.json').read_text())
    micro = json.loads((root/'micro/result.json').read_text())
    stability = json.loads((root/'baseline/stability.json').read_text())
    stable = (stability_summary(root/'baseline') if stability.get('rounds') else
              {'passed': False, 'reason': 'missing stability rounds'})
    quality = json.loads((root/'baseline/quality.json').read_text())
    launcher_path = root/'baseline-launcher-error.json'
    launcher = json.loads(launcher_path.read_text()) if launcher_path.exists() else None
    full_baseline = (base.get('valid') and base.get('released') and base.get('port_released')
                     and len(base['cases']) == 23 and all(len(c['rounds']) >= 5 for c in base['cases'])
                     and stable['passed'] and not quality['execution_errors'])
    for name, expected in (('long-quality.json',72),('boundary-quality.json',72),('prefix-growth.json',68)):
        value = json.loads((root/'baseline'/name).read_text())
        full_baseline = full_baseline and value['passed'] == value['total'] == expected
    grouped = defaultdict(list)
    for row in micro['samples']:
        grouped[(row['projection'],row['M'])].append(row)
    operators = []
    for (projection,m), rows in grouped.items():
        valid = all(r['correct'] for r in rows) and len(rows) == 3
        entry = {'projection':projection,'M':m,'correct_all_three_layers':valid,
                 'selected_kinds': sorted({r.get('kind', 'pipeline' if r.get('pipeline') else 'staged') for r in rows}),
                 'max_relative_l2':max(r['relative_l2'] for r in rows)}
        if valid:
            entry['upstream_ms'] = statistics.median(statistics.median(r['upstream_ms']) for r in rows)
            entry['candidate_ms'] = statistics.median(statistics.median(r['candidate_ms']) for r in rows)
            entry['speed_ratio'] = entry['upstream_ms']/entry['candidate_ms']
        operators.append(entry)
    performance = [{'case': c['name'], 'tps':statistics.median(r['tokens_per_second'] for r in c['rounds']),
                    'ttft_ms':statistics.median(v for r in c['rounds'] for v in r['ttft_ms'])} for c in base['cases']]
    decision = ('needs_model_integration' if micro['passed_micro_gate'] and micro.get('nz_selection') else
                'rejected_no_qualified_native_nz_profile' if micro['passed_micro_gate'] else
                'rejected_graph_capture' if any('graph_error' in r for r in micro['samples']) else
                'rejected_numerical' if not micro['correct'] else 'rejected_no_stable_operator_speedup')
    result = {'baseline_complete':bool(full_baseline),'baseline_launcher_exception':launcher,
              'candidate_decision':decision,
              'end_to_end_candidate_tested':False,'baseline_performance':performance,
              'baseline_quality':quality['categories'],'stability_seconds':stability['elapsed_seconds'],
              'stability_acceptance':stable,
              'operator_comparison':operators,'micro_gate':micro['passed_micro_gate'],
              'qualified_profiles':micro.get('qualified_profiles', []),
              'nz_selection':micro.get('nz_selection', []),
              'limitations':['候选包含分阶段及混合AIV/AIC双缓冲实现，均经GM交换反量化数据。',
                             '算子汇总使用每层最佳配置；模型接入仅接受三层共同通过的同一配置。',
                             '单算子候选未通过时不进入模型接入，不报告候选服务TPS或宣称完整服务加速。']}
    (root/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    lines = ['# 当前优化版基线与AscendC W4A16原型对照','',
             f'完整基线验收：{bool(full_baseline)}；候选结论：{decision}。',
             '','候选包含分阶段/Split-K和混合AIV/AIC双缓冲，使用相同权重与量化语义。',
             '候选服务端到端性能尚未测量；下列算子速度比不得解释为服务加速比。',
             '','## 当前服务基线','', '| 场景 | TPS | TTFT/ms |','|---|---:|---:|']
    if launcher:
        lines[4:4] = ['启动器收尾曾退出127：运行中修改脚本造成Bash读取错位。'
                      'CLI测试已完成并释放引擎和端口；异常原文保留在baseline-launcher-error.json，'
                      '正式服务恢复另行核验。', '']
    lines += [f"| {p['case']} | {p['tps']:.3f} | {p['ttft_ms']:.3f} |" for p in performance]
    lines += ['', '## 真实FFN算子对照', '', '| 投影 | M | 三层数值通过 | 原算子/ms | 候选/ms | 原算子/候选 |','|---|---:|---|---:|---:|---:|']
    for p in operators:
        times = (f"{p['upstream_ms']:.4f} | {p['candidate_ms']:.4f} | {p['speed_ratio']:.3f}" if p['correct_all_three_layers'] else '未接受 | 未接受 | 未接受')
        lines.append(f"| {p['projection']} | {p['M']} | {p['correct_all_three_layers']} | {times} |")
    lines += ['', '## 质量与稳定性', '', json.dumps(quality['categories'],ensure_ascii=False),
              f"连续稳定性时间：{stability['elapsed_seconds']:.1f}秒。", '',
              '逐题结果、逐轮HTTP/SSE、内存采样、各分块配置及精度误差保留在相邻原始JSON/JSONL文件中。']
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    print(json.dumps(summarize(parser.parse_args().root),ensure_ascii=False,indent=2))

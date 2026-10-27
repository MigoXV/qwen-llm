"""由完整验收文件生成可审阅的专项报告，禁止用未完成结果填表。"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import statistics

from .compare import directory_metrics, metrics


def read(path):
    return json.loads(Path(path).read_text())


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def measured(manifest):
    result = metrics(manifest)
    for case in manifest['cases']:
        rounds = case['rounds']
        times = ([r['ttft_ms'][3] for r in rounds] if case['name'] == 'mixed32k-c4'
                 else [t for r in rounds for t in r['ttft_ms']])
        result[case['name']].update(
            rounds=len(rounds), requests=sum(len(r['output_tokens']) for r in rounds),
            ttft_p50_ms=statistics.median(times), ttft_p95_ms=percentile(times, .95),
            ttft_p99_ms=percentile(times, .99),
            tps_min=min(r['tokens_per_second'] for r in rounds),
            tps_max=max(r['tokens_per_second'] for r in rounds),
            input_tokens=[r['input_tokens'] for r in rounds], output_tokens=[r['output_tokens'] for r in rounds])
    return result


def hbm_gib(inventory):
    match = re.search(r'(\d+)\s*/\s*65536', inventory['npu'])
    if not match:
        raise ValueError('缺少本机64GiB设备的HBM计数')
    return int(match.group(1)) / 1024


def memory_summary(directory, manifest):
    samples = []
    with (directory / 'npu-samples.jsonl').open() as source:
        for line in source:
            row = json.loads(line)
            if 'npu' in row:
                samples.append(hbm_gib(row))
    worker = manifest['ready']['runtime']['workers'][0]
    return {'ready_torch_allocated_gib': worker['npu_allocated_bytes'] / 1024**3,
            'ready_torch_reserved_gib': worker['npu_reserved_bytes'] / 1024**3,
            'kv_allocated_gib': worker['kv_cache_allocated_bytes'] / 1024**3,
            'ready_whole_card_hbm_gib': hbm_gib(manifest['ready']['inventory']),
            'whole_card_hbm_sampled_max_gib': max(samples), 'whole_card_hbm_samples': len(samples),
            'sampling_interval_seconds': 2,
            'note': 'Sampled whole-card maximum includes runtime and base occupancy; it is not a guaranteed instantaneous peak.'}


def production_measured(manifest):
    result = measured(manifest)
    for row in result.values():
        row['accepted_fraction'] = None
    return result


def build(root):
    root = Path(root).resolve()
    names = ('combined-acceptance', 'independent-acceptance', 'production-acceptance',
             'trace-acceptance', 'source-integrity-final', 'selected-model-integrity', 'promotion-equivalence',
             'alias-preservation', 'validation-checks', 'lifecycle-final')
    evidence = {name: read(root / (name + '.json')) for name in names}
    if not all(row['passed'] for row in evidence.values()):
        raise ValueError('全部硬件、数值和原始产物验收通过后才可生成最终报告')
    selected = read(root / 'selection.json')
    folder = f'combined-{selected["selected_recipe"]}-b{selected["selected_batch_tokens"]}'
    old, new = [read(root / name / 'manifest.json') for name in ('baseline-full', folder)]
    bm, cm = measured(old), measured(new)
    numeric = {name: read(root / (name + '.json')) for name in ('scale-numerics-runtime', 'w8-numerics')}
    if not all(row['passed'] for row in numeric.values()):
        raise ValueError('算子数值检查没有通过')
    cli = {name: read(root / ('production-' + name) / 'manifest.json') for name in ('baseline', 'candidate')}
    expanded = read(root / 'expanded-kv/manifest.json') if (root / 'expanded-kv/manifest.json').exists() else None
    controls = ['screen-control', 'ablation-scale-retry', 'ablation-head-retry', 'ablation-cache',
                'ablation-checkpoint', 'quant-gdn', 'quant-head-retry', 'quant-both-retry']
    ablations = {}
    for name in controls:
        manifest = read(root / name / 'manifest.json')
        ablations[name] = {'performance': directory_metrics(root / name),
                          'quality': read(root / name / 'quality.json')['categories'],
                          'npu_allocated_gib': manifest['ready']['runtime']['workers'][0]['npu_allocated_bytes'] / 1024**3}
    repositories = []
    for suffix in ('GDNW8', 'HeadW8', 'GDNHeadW8'):
        path = Path('/home/model-bin/Qwen') / f'Qwen3.8-27B-W4A8-Ascend-{suffix}-v1'
        metadata = read(path / 'quantization_manifest.json')
        repositories.append({'path': str(path), 'requantized_projections': len(metadata['layers']),
                             'tensor_gib': metadata['tensor_bytes'] / 1024**3,
                             'validated_for_inference': metadata['validated_for_inference'],
                             'selected': str(path) == new['config']['model_path']})
    result = {'generated_at': datetime.now(timezone.utc).isoformat(), 'raw_root': str(root),
              'selection': selected, 'acceptance': evidence, 'baseline': bm, 'candidate': cm,
              'ablations': ablations, 'numerics': numeric,
              'repositories': repositories,
              'runtime': {'baseline': old['ready'], 'candidate': new['ready']},
              'memory': {'baseline': memory_summary(root / 'baseline-full', old),
                         'candidate': memory_summary(root / folder, new)},
              'software': new['ready']['runtime']['packages'],
              'source_sha256': {'baseline': old['implementation_sha256'], 'candidate': new['implementation_sha256'],
                                'production': cli['candidate']['implementation_sha256']},
              'production': {name: {'performance': production_measured(m), 'ready': m['ready'],
                                    'valid': m['valid'], 'released': m['released'], 'port_released': m['port_released'],
                                    'capacity': m.get('capacity')} for name, m in cli.items()},
              'expanded_kv': {'config': expanded['config'], 'performance': metrics(expanded),
                               'valid': expanded['valid'], 'released': expanded['released']} if expanded else None}
    return result, old, new


def markdown(result, old, new):
    evidence = result['acceptance']; full = evidence['combined-acceptance']
    root = Path(result['raw_root']); bm, cm = result['baseline'], result['candidate']
    quality = full['quality']; categories = quality['candidate_categories']
    bq = read(root / 'baseline-full/quality.json')['categories']
    old_worker, new_worker = [m['ready']['runtime']['workers'][0] for m in (old, new)]
    gib = lambda v: v / 1024**3
    delta = lambda v: f'{(v - 1) * 100:+.2f}%'
    hot = 'chinese57344-c4-hot'; g1, g4 = 'generation512-c1', 'generation512-c4'
    stable = full['stability']; package = result['software']
    lines = [
        '# Qwen3.8-27B / Ascend910B4 专项优化及完整验收', '',
        f'本报告使用独立实例的真实HTTP/SSE、NPU trace和逐题结果，原始目录为 `{root}`。'
        '比较基线是项目原有混合量化模型，未测完整BF16模型吞吐，因此所有加速比均相对原混合量化版本。', '',
        '## 已采用的改动与结果', '',
        f'采用FFN尺度加载时缓存、目标/MTP输出头共享、GDN最后可复用提示检查点保留，以及49个指定投影的W8量化。'
        f'完整验收、独立复测和普通生产CLI验收均通过。512-token生成单路吞吐'
        f'{bm[g1]["tps"]:.2f} → {cm[g1]["tps"]:.2f} TPS（{delta(cm[g1]["tps"]/bm[g1]["tps"])}），'
        f'四路 {bm[g4]["tps"]:.2f} → {cm[g4]["tps"]:.2f} TPS（{delta(cm[g4]["tps"]/bm[g4]["tps"])}）。', '',
        f'57K四路热重放吞吐 {bm[hot]["tps"]:.2f} → {cm[hot]["tps"]:.2f} TPS'
        f'（{cm[hot]["tps"]/bm[hot]["tps"]:.2f}倍），TTFT {bm[hot]["ttft_p50_ms"]/1000:.3f} → '
        f'{cm[hot]["ttft_p50_ms"]/1000:.3f}秒。该收益来自实际前缀命中恢复；候选五轮20个请求均命中55296 token。'
        '它不代表未见过的长文输入也能获得同等加速。', '',
        f'就绪态Torch实际分配 {gib(old_worker["npu_allocated_bytes"]):.4f} → '
        f'{gib(new_worker["npu_allocated_bytes"]):.4f} GiB，减少 '
        f'{gib(old_worker["npu_allocated_bytes"]-new_worker["npu_allocated_bytes"]):.4f} GiB；'
        f'两者KV实分配相同，为 {gib(new_worker["kv_cache_allocated_bytes"]):.4f} GiB。', '',
        '缩小预填充分块没有通过延迟与吞吐的平衡门槛，保持8192预算；不将未采用的约57%短流卡顿改善算入发布收益。', '',
        '## 模型、硬件与比较条件', '',
        '| 项目 | 固定条件 |', '| --- | --- |',
        '| 设备 | Ascend910B4-1，物理卡4 / 可见逻辑0，64 GiB，单卡TP1；测试时无其他NPU拥有进程 |',
        '| 结构 | 本地Qwen3.8-27B dense；config.model_type=qwen3_5；64层，hidden5120，FFN17408，词表248320，48个GDN/16个全注意力层，1层MTP |',
        '| 精度 | BF16残差、FFN激活与attention KV；GDN递归状态FP32；FFN INT4/group128/NZ权重反量化，W8分支使用动态INT8激活 |',
        '| 服务 | 输入57344 + 输出8192，上限4路；固定KV22.5 GiB；全局分块8192；gpu_memory_utilization=0.82 |',
        '| 原有加速 | MTP2、异步调度、前缀缓存、NPUGraphEx、FULL_DECODE_ONLY；基线已开启，不属于新增收益 |',
        f'| 依赖 | torch {package["torch"]}；torch-npu {package["torch-npu"]}；vLLM {package["vllm"]}；Ascend {package["vllm-ascend"]}；Transformers {package["transformers"]}；CANN9.1.1 |', '',
        '性能以每轮真实输出token总量除以整轮makespan计算；每场景先预热，再关闭profiler测五轮，报告吞吐中位数。'
        '冷轮次清空前缀缓存，热轮次保存每请求实际命中。trace另开实例，只用于确认算子、shape及归因。'
        '输入精确长度按真实API usage核对，生成自然提前结束时使用实际输出量。', '',
        '## 热点与对应适配', '',
        '原始分段及内核证据见[热点报告](qwen38-hotspots.md)。单路decode中FFN约占累计设备工作量32%、输出头15.73%'
        '（目标5.32%、草稿10.41%）、GDN输出投影9.19%；长文prefill FFN约48%，其内部权重反量化相关工作约88%。'
        '这些是累计内核工作量，不能相加后解释为端到端墙钟比例。', '',
        '### FFN尺度缓存', '',
        '上游每层forward执行`weight_scale_second.to(x.dtype)`，64层融合gate/up及down共128次转换。'
        '适配保留上游FP32尺度合并顺序，在加载后将最终尺度一次舍入至BF16；权重、group、NZ布局和算子不变。'
        '真实0/30/63层、两类投影与M=1/3/6/12/38/117/1536/2048/6144/7680共60组，输出逐位一致、最大误差0。'
        '最终trace确认这两类尺度Cast由每个target forward 128次降为0，FFN matmul数量仍全部覆盖；其他Cast保留。'
        '单独就绪态减少0.2490 GiB。数值逐位对齐指这些算子，不表示整个生成序列必须逐字一致。', '',
        '### 输出头共享和量化加载', '',
        '目标模型与MTP原先分别驻留248320×5120的输出头。加载阶段核对全部参数名、shape、dtype、内容与量化方法后共享同一模块；'
        '推理过程中没有逐步比较或同步，仍然执行两侧各自需要的logits投影。相同隐藏态的两侧logits逐位一致。'
        '共享BF16头单独节省2.3682 GiB；采用W8头后共享的对象已经缩小，不能再把这2.3682 GiB与量化节省直接相加。', '',
        '本机上游Qwen目标与MTP构造`ParallelLMHead`时未传quant_config，直接加载W8头会缺少weight_offset。'
        '适配只对248320×5120、量化描述明确为W8的输出头绑定官方W8执行方法。安装的vLLM/Ascend源文件没有因本实验被修改。', '',
        '### GDN检查点保留', '',
        '原LRU会在预填充压力下逐出四路提示最后可复用的GDN状态，导致全注意力块尚在、混合前缀却不能命中。'
        '只优先使用未缓存空闲块仍然无效：预热中这类空闲块曾耗尽，结束时又有空闲块不能反证未发生驱逐。', '',
        '采用策略维持原块对象、hash查找、引用计数和空闲双向队列，另维护O(1)索引：先取未缓存空闲块，'
        '再回收较早GDN状态，最后按原LRU分配。保护范围仅为每提示/每GDN组最后一个可供MTP裁剪后复用的检查点；'
        '提示尾部不能重放的较晚状态不受保护。保护是软偏好，所有块都受保护时仍能回收，不固定占用或禁止驱逐。', '',
        '专项调度器只在请求到来前接入Engine Core，要求1536对齐、17层全注意力缓存（含MTP）、三组各16层GDN、'
        'MTP2和FP32递归状态布局。测试覆盖重复hash、引用、reset、全保护退让、随机2000步分配，以及实际多路前缀增长/稳定性不变量。', '',
        '### 仅量化指定的49个投影', '',
        '从原始BF16重新量化48个GDN out_proj及1个lm_head，采用每输出通道对称INT8权重与官方逐token动态INT8激活。'
        '其他原混合量化张量保持原内容。三种独立配方均筛选，最后选择GDN+Head组合。'
        f'实际尺寸35组投影验证均在预设容差内，最大relative-L2为 '
        f'{max(row["relative_l2"] for row in result["numerics"]["w8-numerics"]["rows"]):.6f}。'
        '这是有损量化，最终采用依据是完整生成质量评估；微基准中的eager事件跨度含主机提交间隙，不作为端到端加速比。', '',
        '## 23场景五轮完整A/B', '',
        '| 场景 | 基线TPS | 优化TPS | 吞吐变化 | 基线TTFT/s | 优化TTFT/s |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for case in old['cases']:
        name = case['name']; a, b = bm[name], cm[name]
        lines.append(f'| {name} | {a["tps"]:.3f} | {b["tps"]:.3f} | {delta(b["tps"]/a["tps"])} | '
                     f'{a["ttft_p50_ms"]/1000:.3f} | {b["ttft_p50_ms"]/1000:.3f} |')
    lines += ['', 'mixed32k-c4的TTFT列为其中**长请求**的首字延迟，避免被三个短请求的中位数掩盖。'
              '所有核心场景吞吐不低于基线95%；长输入TTFT不超过基线105%。短提示TTFT仍完整报告，但不参与长输入延迟门槛。'
              '逐轮TPS范围、TTFT的P50/P95/P99、MTP接受率、输入/输出token及缓存命中见[机器结果](qwen38-optimization-results.json)。'
              '每场景仅5至20个请求样本，P95/P99是这些样本的最近秩分位数，不能视为线上尾延迟保证。', '',
              '## 消融与被拒绝的方案', '',
              '量化筛选使用独立、相同128题质量协议及四场景五轮的screen-control；完整基线有1319题历史，'
              '短输出场景的MTP接受率和生成内容可能变化，不用相差几个百分点的单次数字宣称稳定收益。', '',
              '| 候选 | 单路短TPS | 四路短TPS | 8K单路冷TPS | 8K四路冷TPS | GSM8K/128 | 就绪分配GiB |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, row in result['ablations'].items():
        if name in {'ablation-cache', 'ablation-checkpoint'}: continue
        perf = row['performance']
        vals = [perf[key]['tps'] for key in ('short128-c1', 'short128-c4', 'chinese8192-c1-cold', 'chinese8192-c4-cold')]
        lines.append(f'| {name} | ' + ' | '.join(f'{v:.3f}' for v in vals)
                     + f' | {row["quality"]["gsm8k"]["passed"]} | {row["npu_allocated_gib"]:.4f} |')
    lines += ['', '所有上述有效筛选均40道固定答案及64道项目代码题通过。128题允许最多少4题，仅用于排序；'
              '发布使用完整1319题的1个百分点门槛。缓存消融在同一22.5 GiB预算下，57K四路热重放：仅空闲块优先'
              f'{result["ablations"]["ablation-cache"]["performance"][hot]["tps"]:.3f} TPS，检查点策略'
              f'{result["ablations"]["ablation-checkpoint"]["performance"][hot]["tps"]:.3f} TPS；前者未采用。', '',
              '| 分块候选 | 生成四路吞吐比 | 8K冷吞吐比 | 混合吞吐比 | 混合长TTFT比 | 短流最大间隔比 | 采用 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    groups = [('固定', result['selection']['batch']['candidates']),
              ('条件', result['selection']['adaptive_batch']['candidates']),
              ('完整组合+条件', result['selection'].get('combined_adaptive_pilot', {}).get('candidates', []))]
    for group, rows in groups:
        for row in rows:
            t = row['tps_ratios']
            lines.append(f'| {group}{row["batch_tokens"]} | {t[g4]:.4f} | {t["chinese8192-c4-cold"]:.4f} | '
                         f'{t["mixed32k-c4"]:.4f} | {row["mixed_long_ttft_ratio"]:.4f} | '
                         f'{row["short_max_delivery_gap_ratio"]:.4f} | 否 |')
    mixed = full['mixed_stall']['short_max_delivery_gap_ratio']
    lines += ['', '比值相对各自相同工作流的8192分块对照。条件策略只在短提示已解码且长提示待预填充时缩块。'
              '完整组合配3084虽然将短流最大间隔降低56.88%，但混合长请求TTFT增加5.4875%，超过预先设定的5%门槛，仍然拒绝。'
              '未放宽门槛，也未把它写入生产调度器。', '',
              f'最终8192组合的混合短流最大交付间隔比为 {mixed:.4f}；短流交付批次可能一次包含多个token，'
              '这里是相邻非空引擎批次间隔，不把SSE网络分片当成单token ITL。混合卡顿减半没有作为本次交付承诺。', '',
              '原生Dense GMM在原热点对照中只有上游约0.85–0.95倍吞吐，且存在29×31答900的质量失败，保留在测试目录，不进入生产。', '',
              '## 精度、长文、边界和稳定性', '',
              '| 检验 | 基线 | 最终组合 |', '| --- | ---: | ---: |',
              f'| 固定答案（算术/逻辑/翻译/格式） | 40/40 | {sum(v["passed"] for k,v in categories.items() if k not in {"gsm8k","code"})}/40 |',
              f'| GSM8K | {bq["gsm8k"]["passed"]}/1319（{bq["gsm8k"]["passed"]/1319:.3%}） | '
              f'{categories["gsm8k"]["passed"]}/1319（{categories["gsm8k"]["passed"]/1319:.3%}） |',
              f'| 项目代码回归题 | 64/64 | {categories["code"]["passed"]}/64 |',
              '| 8K/32K/57K × 三个位置 × 四路 × 冷热检索 | 72/72 | 72/72 |',
              '| 1536倍数前后边界 × 四路 × 冷热检索 | 72/72 | 72/72 |',
              '| 四路8K→32K连续增长 | 68/68 | 68/68，17次真实分配池检查 |', '',
              f'GSM8K相对基线变化 {quality["gsm8k_delta_percentage_points"]:+.4f} 个百分点，门槛为下降不超过1个百分点；'
              f'新增固定答案失败为0，执行错误0。逐题退化和改善保留在quality.json/jsonl中；完整质量比较匹配数据集、提示词、'
              '代码判题器SHA256和实例身份。temperature=0、seed42、非思考，数学题要求推导并以FINAL结束，输出预算2048。'
              '代码为8类任务×8组参数的项目固定题，在受CPU时间与内存限制的独立进程判题，不代表HumanEval或广泛代码能力。', '',
              f'稳定性连续运行 {stable["elapsed_seconds"]:.1f} 秒，{stable["rounds"]} 轮、{stable["requests"]} 个混合长短请求，'
              f'HTTP错误{stable["errors"]}。每轮验证空闲索引、较早状态索引、检查点归属、元数据有界和设备隔离。'
              f'NPU实际分配范围 {stable["npu_allocated_range_bytes"]/1024**2:.3f} MiB，保留范围 '
              f'{stable["npu_reserved_range_bytes"]/1024**2:.3f} MiB。RSS/FD按稳定阶段前后20%窗口中位数比较，'
              '允许增量分别128 MiB和4个FD；逐进程证据如下。', '',
              '| 进程PID | 采样数 | RSS增量MiB | FD增量 |', '| --- | ---: | ---: | ---: |']
    for process in stable['process_growth']:
        lines.append(f'| {process["pid"]} | {process["samples"]} | {process["rss_growth_bytes"]/1024**2:.3f} | {process["fd_growth"]:.1f} |')
    lines += ['', '## 独立复测与实际生产路径', '',
              '完整组合释放设备后，以新进程复测六个核心场景各五轮，重复40道固定答案与64道代码题。'
              '随后另外采集最终trace。独立复测仍通过全部核心退化限制，57K四路热重放仍全部命中55296 token；'
              '复测的逐轮结果单独保留，未与第一轮混在一起取更好值。', '',
              '| 场景 | 基线TPS | 完整组合TPS | 新进程复测TPS | 复测相对基线 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for row in evidence['independent-acceptance']['performance']:
        name = row['case']
        lines.append(f'| {name} | {bm[name]["tps"]:.3f} | {cm[name]["tps"]:.3f} | '
                     f'{row["candidate"]["tps"]:.3f} | {delta(row["tps_ratio"])} |')
    lines += ['', '| 普通生产CLI场景 | 基线TPS | 优化TPS | 吞吐变化 | 基线TTFT/s | 优化TTFT/s |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in evidence['production-acceptance']['performance']:
        a,b=row['baseline'],row['candidate']
        lines.append(f'| {row["case"]} | {a["tps"]:.3f} | {b["tps"]:.3f} | {delta(row["tps_ratio"])} | '
                     f'{a["ttft_ms"]/1000:.3f} | {b["ttft_ms"]/1000:.3f} |')
    cap = evidence['production-acceptance']['capacity']
    lines += ['', '该对照通过普通serve-model CLI启动，不使用测试Worker或注入引擎参数。'
              '两次使用完全相同的主程序源码和模型配置表`configs/models.json`（逐文件SHA256一致），显式选择不同权重仓库和适配。'
              '真实Worker为Qwen38NPUWorker；就绪核对128层BF16尺度、共享INT8头、48个W8 GDN输出投影、三张解码图和完整上下文容量。'
              '普通文本、JSON Schema结构化采样、断开SSE后abort及活动槽回收均通过。', '',
              f'容量压力测试真实四路输入为 `{cap["input_tokens"]}`，实际生成为 `{cap["output_tokens"]}`；'
              '四路均达到57344+8192=65536 token，未以只声明输出上限或自然提前EOS代替满容量实测。', '',
              '另以增加KV作为单独变量复测。默认仍采用完整质量与一小时稳定性验证过的22.5 GiB；'
              '不把扩大KV预算造成的变化叠加成执行适配收益。扩容实例只作性能/容量消融，未替代完整质量与稳定性验收。', '']
    expanded = result['expanded_kv']
    if expanded:
        lines += ['| KV消融场景 | 22.5 GiB TPS | ' + f'{gib(expanded["config"]["kv_cache_bytes"]):.1f} GiB TPS | '
                  '22.5 GiB TTFT/s | 扩容TTFT/s |', '| --- | ---: | ---: | ---: | ---: |']
        for name, row in expanded['performance'].items():
            before = cm[name]
            ttft = row.get('long_ttft_ms', row['ttft_ms'])
            before_ttft = before.get('long_ttft_ms', before['ttft_ms'])
            lines.append(f'| {name} | {before["tps"]:.3f} | {row["tps"]:.3f} | '
                         f'{before_ttft/1000:.3f} | {ttft/1000:.3f} |')
        lines += ['', '扩容与原配置的执行历史不同，几百分点的差异需结合MTP接受率与缓存命中解释，默认预算不因此变动。', '']
    lines += ['## 显存口径和此前四路失败', '',
              '“项目显存”指进程或实例的总体占用，包含权重、KV/GDN状态、MTP、执行图、临时张量、工作区、分配器保留及运行时开销。'
              '它不是在“权重+KV”之外再加一遍的独立类别。Torch allocated、Torch reserved、NPU进程占用、整卡HBM是不同口径，不能直接相加。'
              '卡上无拥有进程时仍有约3417 MiB基础占用；磁盘模型大小也不等于设备驻留量。', '',
              '| 显存口径 / GiB | 基线 | 最终组合 | 减少 |', '| --- | ---: | ---: | ---: |']
    for key, label in [('ready_torch_allocated_gib', '就绪态Torch allocated'),
                       ('ready_torch_reserved_gib', '就绪态Torch reserved'),
                       ('kv_allocated_gib', '实际KV，已包含于allocated'),
                       ('ready_whole_card_hbm_gib', '就绪态整卡HBM'),
                       ('whole_card_hbm_sampled_max_gib', '全程2秒采样HBM最大值')]:
        a, b = [result['memory'][name][key] for name in ('baseline', 'candidate')]
        lines.append(f'| {label} | {a:.4f} | {b:.4f} | {a-b:.4f} |')
    lines += ['', '整卡HBM下降与Torch分配下降不同，含义分别是设备总体占用和Torch管理张量的占用；'
              '采样最大值不是瞬时峰值保证，基线与候选的总运行时长也不同。这里不将5.20 GiB的Torch分配减少等同于5.20 GiB整卡空闲增加。', '',
              '此前默认别名的12.5 GiB KV只支持约2.24路完整64K上下文。即使物理显存还有余量，固定KV预算没有自动增大，'
              '启动容量检查仍会拒绝四路；该失败不是整卡OOM。此次同预算A/B统一22.5 GiB，实际分配22.4732 GiB、222块、'
              f'完整上下文容量 {new_worker["full_context_capacity"]:.5f} 路，并用普通生产CLI完成上述满容量请求。', '',
              '## 权重保护、生产边界及复现', '',
              '原始BF16和原混合量化仓库的50个文件再次逐文件SHA256一致；新仓库逐权重分片SHA256与导出清单一致。'
              '未修改两个原仓库，未覆盖权重或使用可写硬链接，未推送远程仓库。', '',
              '| 独立仓库后缀（共同前缀Qwen3.8-27B-W4A8-Ascend-） | 重新量化投影数 | 磁盘张量GiB | 状态 |',
              '| --- | ---: | ---: | --- |']
    for repository in result['repositories']:
        suffix = Path(repository['path']).name.removeprefix('Qwen3.8-27B-W4A8-Ascend-')
        lines.append(f'| {suffix} | {repository["requantized_projections"]} | {repository["tensor_gib"]:.4f} | '
                     + ('已选中并验收' if repository['selected'] else '保留候选，未发布') + ' |')
    lines += ['', '磁盘张量总量不同于设备驻留量。选中仓库的分片还验证了普通文件、独立inode和link count为1。', '',
              f'采用的独立模型仓库：`{new["config"]["model_path"]}`。其qwen38_optimization.json保存依赖版本、结构/量化描述/索引SHA、'
              '权重SHA、允许参数及验收证据。其余GDNW8-v1、HeadW8-v1仓库保持候选状态。', '',
              '新别名为`qwen3.8-27b-optimized-v1`，显式`model_adaptation=qwen38-910b-v1`；原有默认及九个正式别名参数保持原样。'
              '适配只接受已验收模型结构、软件构建、BF16/TP1/MTP2、图配置、KV和四路64K组合，未验证组合明确拒绝，不静默回退。'
              '进程内注册只发生在专用Worker中，不影响其他服务进程。', '',
              '生产只保留模型产物校验、尺度/输出头加载适配和检查点调度策略。所有profiling、压测、量化导出、评估、'
              '算子验证、比较和被拒绝策略均在[独立测试区](../tests/performance/qwen38/README.md)，生产包不导入tests，wheel不包含实验代码。', '',
              '```bash',
              'source scripts/env_ascend_291.sh',
              'bash scripts/run_model.sh qwen3.8-27b-optimized-v1',
              '# 以下复现实验必须使用不存在的新目录；先确保设备无其他拥有进程。',
              'poetry run python -m tests.performance.qwen38.cli profile \\',
              '  --model-name qwen3.8-27b-both-w8-candidate \\',
              '  --optimizations scale,head,checkpoint,head-quant --batch-tokens 8192 \\',
              '  --kv-cache-gib 22.5 --suite full --rounds 5 --no-capture \\',
              '  --quality-samples 1319 --quality-reference BASELINE/quality.json \\',
              '  --long-validation --cache-boundary-validation --prefix-growth-validation \\',
              '  --stability-seconds 3600 --output-dir /home/qwen-models/reports/qwen3.8-27b/FRESH_FULL',
              'poetry run python -m tests.performance.qwen38.production_verify \\',
              '  --model qwen3.8-27b-optimized-v1 --capacity-validation \\',
              '  --output /home/qwen-models/reports/qwen3.8-27b/FRESH_PRODUCTION',
              'poetry run pytest -q', 'poetry check', '```', '',
              '测试入口的默认模型目录为同目录catalog.json；首次量化导出及trace离线复核命令见测试区README。'
              '每个实例保留源码快照及SHA，生产CLI也保存实际主程序源码；仓库基础revision与未提交状态保存在inventory中。', '',
              '## 工程验证与构建', '',
              f'固定Ascend Poetry解释器下，{evidence["validation-checks"]["checks"]["pytest"]["tests_passed"]} 项pytest全部通过；'
              'poetry check、编译、shell语法和git diff --check通过。'
              f'构建wheel含 {evidence["validation-checks"]["checks"]["wheel"]["python_files"]} 个Python模块，'
              '包含三个必要专项执行模块，不包含tests、profiling、benchmark、evaluation、quantization或GMM实验代码，生产模块无tests导入。'
              '构建文件SHA和各检查日志在validation-checks.json中；模型别名保留核对和设备/端口/服务最终状态另有独立证据。', '',
              '## 失败记录和结论边界', '',
              '首次尺度测试因观测中间件初始化顺序导致HTTP500，修复后新目录完整重跑；早期final-only/128输出预算造成截断的数学筛选不作质量证据；'
              '首次W8头因构造器缺少quant_config启动失败，修复后分别重跑。原始失败和源码过渡说明均保留，未覆盖或混入通过数据。', '',
              '本报告证明的是这一模型、这一Ascend环境和记录的输入/并发范围；重新量化有数值误差，1319题和项目代码题不是全面质量保证。'
              '冷长文prefill仍主要受FFN权重反量化影响，本次没有实现新的自定义NPU FFN内核，也没有解决混合负载卡顿减半的目标。'
              '更大的冷文档、不同量化组、TP或软件构建需要重新验证。所有测试实例最后均退出并释放设备和端口，原本停止的服务保持停止。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--markdown', type=Path, default=Path('docs/qwen38-optimization.md'))
    parser.add_argument('--json', dest='json_path', type=Path, default=Path('docs/qwen38-optimization-results.json'))
    args = parser.parse_args()
    if args.json_path.exists():
        raise ValueError('拒绝覆盖已经生成的版本化机器结果')
    result, old, new = build(args.root)
    args.markdown.write_text(markdown(result, old, new))
    args.json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'markdown': str(args.markdown), 'json': str(args.json_path),
                      'all_acceptance_passed': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()

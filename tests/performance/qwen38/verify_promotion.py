"""比较完整验收源码快照和生产执行核心，记录迁移未改变已验证算子。"""
import argparse
import ast
import hashlib
import json
from pathlib import Path


MAPPING = {
    'runtime_candidate.py': ('qwen38.py', ('file_sha256', 'model_shape', 'artifact')),
    'runtime_ops_candidate.py': ('qwen38_worker.py', (
        'CachedScaleLinearMethod', 'install_scale_cache', 'install_head_sharing', 'install_quantized_head')),
    'runtime_cache_candidate.py': ('qwen38_cache.py', ('Qwen38BlockPool',)),
}


def definitions(path):
    tree = ast.parse(path.read_text())
    return tree, {node.name: node for node in tree.body
                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def verify(root, project):
    selection = json.loads((root / 'selection.json').read_text())
    folder = f'combined-{selection["selected_recipe"]}-b{selection["selected_batch_tokens"]}'
    source = root / folder / 'source/tests/performance/qwen38'
    production = project / 'src/qwen_llm/engine'
    rows = []; files = {}
    for candidate_name, (production_name, names) in MAPPING.items():
        old_path, new_path = source / candidate_name, production / production_name
        _, old = definitions(old_path)
        tree, new = definitions(new_path)
        for name in names:
            equal = ast.dump(old[name]) == ast.dump(new[name])
            rows.append({'candidate': str(old_path), 'production': str(new_path),
                         'definition': name, 'syntax_tree_identical': equal})
        files.update({str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (old_path, new_path)})
        if production_name == 'qwen38_cache.py':
            bases = new['Qwen38CheckpointScheduler'].bases
            rows.append({'definition': 'Qwen38CheckpointScheduler bases',
                         'syntax_tree_identical': len(bases) == 1 and isinstance(bases[0], ast.Name)
                         and bases[0].id == 'AsyncScheduler'})
            left_init = next(n for n in old['Qwen38CheckpointScheduler'].body
                             if isinstance(n, ast.FunctionDef) and n.name == '__init__')
            right_init = next(n for n in new['Qwen38CheckpointScheduler'].body
                              if isinstance(n, ast.FunctionDef) and n.name == '__init__')
            for node in ast.walk(right_init):
                if isinstance(node, ast.ImportFrom) and node.module == 'qwen38' and node.level == 1:
                    node.module = 'runtime_candidate'
            rows.append({'definition': 'Qwen38CheckpointScheduler layout validation and pool adoption',
                         'syntax_tree_identical': ast.dump(left_init) == ast.dump(right_init)})
        text = new_path.read_text()
        rows.append({'definition': production_name + ' excludes failed policy and test routing',
                     'syntax_tree_identical': all(word not in text for word in (
                         'mixed_prefill_tokens', 'ShortDecodeBudget', 'has_short_decode_and_long_prefill',
                         'tests.performance', 'ProfilingNPUWorker', 'diagnostic_'))})
    imports = []
    for path in (project / 'src/qwen_llm').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imports += [str(path) + ':' + alias.name for alias in node.names if alias.name.startswith('tests')]
            if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('tests'):
                imports.append(str(path) + ':' + node.module)
    return {'passed': not imports and all(row['syntax_tree_identical'] for row in rows),
            'definitions': rows, 'production_test_imports': imports, 'files_sha256': files,
            'scope': 'AST equality of executed scale/head methods, artifact shape checks and BlockPool; '
                     'new public worker/CLI integration is separately validated by the ordinary production CLI.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--project', type=Path, default=Path.cwd())
    args = parser.parse_args()
    output = args.root / 'promotion-equivalence.json'
    if output.exists():
        raise ValueError('拒绝覆盖生产迁移的源码比较记录')
    result = verify(args.root.resolve(), args.project.resolve())
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

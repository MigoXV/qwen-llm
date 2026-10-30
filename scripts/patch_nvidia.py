"""只为固定版本的 CUDA 环境应用已验证补丁，拒绝覆盖未知源码。"""
import hashlib
from importlib.metadata import version
from importlib.util import find_spec
import json
from pathlib import Path
import subprocess


def main():
    patches = Path(__file__).resolve().parents[1] / 'deploy/nvidia/patches'
    for name in ('shm-notify-conflate', 'qwen35-embedding-quant'):
        apply_patch(patches, name)


def apply_patch(patches, name):
    manifest = json.loads((patches / (name + '.json')).read_text())
    if version('vllm') != manifest['vllm_version']:
        raise RuntimeError('NVIDIA 补丁只验证了 vLLM ' + manifest['vllm_version'])
    package = Path(find_spec('vllm').origin).parent
    source = package / manifest['file']
    digest = lambda: hashlib.sha256(source.read_bytes()).hexdigest()
    if digest() == manifest['after_sha256']:
        print(f'NVIDIA 补丁 {name} 已存在')
        return
    if digest() != manifest['before_sha256']:
        raise RuntimeError('vLLM 源码哈希未知，拒绝覆盖；请先审查现有修改')
    command = ['patch', '--batch', '--forward', '-p1', '-d', str(package),
               '-i', str(patches / manifest['patch'])]
    subprocess.run(command + ['--dry-run'], check=True)
    subprocess.run(command, check=True)
    if digest() != manifest['after_sha256']:
        raise RuntimeError('补丁结果哈希不匹配')
    print(f'NVIDIA 补丁 {name} 已应用，需重启实例生效')


if __name__ == '__main__':
    main()

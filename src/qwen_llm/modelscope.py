"""从 ModelScope 下载公开权重，并核对平台提供的大小及 SHA256。"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
from pathlib import Path, PurePosixPath
import subprocess

import httpx

logger = logging.getLogger(__name__)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download_modelscope(model_id: str, output_dir: Path, workers: int = 4) -> None:
    if len(model_id.split('/')) != 2:
        raise ValueError('模型 ID 必须为 owner/name')
    response = httpx.get(f'https://modelscope.cn/api/v1/models/{model_id}/repo/files',
                         params={'Revision': 'master', 'Recursive': 'true'}, timeout=60,
                         follow_redirects=True)
    response.raise_for_status()
    listing = response.json()
    if not listing.get('Success'):
        raise RuntimeError(f'ModelScope 文件清单读取失败：{listing.get("Message")}')
    files = [item for item in listing['Data']['Files'] if item['Type'] == 'blob']
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()
    for item in files:
        path = PurePosixPath(item['Path'])
        if path.is_absolute() or '..' in path.parts or '.git' in path.parts:
            raise ValueError('仓库包含不安全的文件路径')
        if not (root / item['Path']).resolve().is_relative_to(root):
            raise ValueError('下载目标越过模型目录')
    manifest = {'repo_id': model_id, 'listing': listing, 'verified': False}
    manifest_path = root / '.modelscope-manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    def fetch(item):
        target = root / item['Path']
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.stat().st_size == item['Size'] and sha256(target) == item['Sha256']:
                logger.info('已校验，跳过 %s', item['Path'])
                return
            raise RuntimeError(f'目标文件已存在但校验不匹配：{target}')
        partial = target.with_name(target.name + '.part')
        url = f'https://modelscope.cn/models/{model_id}/resolve/{item["Revision"]}/{item["Path"]}'
        logger.info('下载 %s，%.2f GiB', item['Path'], item['Size'] / 1024**3)
        with target.with_name(target.name + '.download.log').open('a') as log:
            subprocess.run(['curl', '--fail', '--location', '--retry', '5', '--retry-delay', '2',
                            '--connect-timeout', '30', '--speed-time', '90', '--speed-limit', '1024',
                            '--continue-at', '-', '--output', str(partial), url],
                           stdout=log, stderr=log, check=True)
        if partial.stat().st_size != item['Size'] or sha256(partial) != item['Sha256']:
            raise RuntimeError(f'下载文件大小或 SHA256 校验失败：{item["Path"]}')
        partial.replace(target)
        logger.info('下载与 SHA256 校验完成：%s', item['Path'])

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(fetch, files))
    manifest['verified'] = True
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    logger.info('全部文件已校验：%s', output_dir)

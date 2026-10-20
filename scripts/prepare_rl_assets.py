"""Prepare and verify RL model, tokenizer, seed and simulator assets."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

REPO = 'RLinf/RLinf-Pi05-RoboTwin-SFT-adjust_bottle'


def fetch(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    content = subprocess.check_output(
        ['curl', '-fsSL', '--retry', '4', '--max-time', '120', url])
    target.write_bytes(content)
    return content


def fetch_large(url, target, total_bytes, expected_sha):
    """Download a large file with curl resume and verify its exact content."""
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        'curl', '--http1.1', '-fL', '--retry', '20', '--continue-at', '-',
        '--speed-time', '120', '--speed-limit', '1024', '--output',
        str(target), url
    ],
                   check=True)
    if target.stat().st_size != total_bytes:
        raise ValueError(f'Invalid download size: {target}')
    with target.open('rb') as stream:
        if digest(stream) != expected_sha:
            raise ValueError(f'Invalid download checksum: {target}')


def tokenizer(dest):
    target = dest / 'paligemma_tokenizer.model'
    expected = ('8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77'
                'a168fc6')
    if target.exists() and hashlib.sha256(
            target.read_bytes()).hexdigest() == expected:
        return
    # The public GCS URL currently resolves through a broken accelerator in
    # this machine. Keep TLS verification on and use the public byte mirror.
    source = ('https://hf-mirror.com/leo009/paligemma_tokenizer.model/'
              'resolve/main/paligemma_tokenizer.model')
    content = fetch(source, target)
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected:
        raise ValueError('Tokenizer SHA256 mismatch')
    (dest / 'tokenizer_source.json').write_text(
        json.dumps(
            {
                'canonical_url': ('https://storage.googleapis.com/big_vision/'
                                  'paligemma_tokenizer.model'),
                'download_url':
                source,
                'sha256':
                actual,
                'reason':
                'GCS TLS hostname mismatch; mirror pinned by content hash'
            },
            indent=2) + '\n')


def weights(root):
    endpoint = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
    dest = root / 'weights' / REPO.split('/')[-1]
    info = json.loads(
        fetch(f'{endpoint}/api/models/{REPO}', dest / 'repo.json'))
    revision = info['sha']
    tree = json.loads(
        fetch(
            f'{endpoint}/api/models/{REPO}/tree/{revision}'
            '?recursive=true&limit=100', dest / 'tree.json'))
    records = []
    for item in tree:
        name = item['path']
        if item['type'] != 'file' or not (
                name.startswith('model') or name == 'config.json'
                or name == 'physical-intelligence/robotwin/norm_stats.json'):
            continue
        target = dest / name
        url = f'{endpoint}/{REPO}/resolve/{revision}/{name}'
        sha = item.get('lfs', {}).get('oid')
        if target.exists() and sha:
            with target.open('rb') as stream:
                actual = hashlib.file_digest(
                    stream, 'sha256').hexdigest() if hasattr(
                        hashlib, 'file_digest') else digest(stream)
            if actual == sha:
                records.append({
                    'path': name,
                    'sha256': sha,
                    'size': target.stat().st_size
                })
                continue
        print(f'Download {name} at {revision}', flush=True)
        if sha and item['size'] > 100_000_000:
            fetch_large(url, target, item['size'], sha)
        else:
            fetch(url, target)
        with target.open('rb') as stream:
            actual = digest(stream)
        if sha and actual != sha:
            raise ValueError(f'SHA256 mismatch: {name}')
        records.append({
            'path': name,
            'sha256': actual,
            'size': target.stat().st_size
        })
    tokenizer(dest)
    tokenizer_path = dest / 'paligemma_tokenizer.model'
    records.append({
        'path':
        tokenizer_path.name,
        'sha256':
        hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    })
    (dest / 'manifest.json').write_text(
        json.dumps({
            'repository': REPO,
            'revision': revision,
            'files': records
        },
                   indent=2) + '\n')
    print('WEIGHTS_READY', dest, flush=True)


def digest(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        result.update(block)
    return result.hexdigest()


def checkout(root):
    target = root / 'src' / 'RoboTwin'
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        subprocess.run([
            'git', 'clone', '--depth', '1', '--branch', 'RLinf_support',
            'https://github.com/RoboTwin-Platform/RoboTwin.git',
            str(target)
        ],
                       check=True)
    commit = subprocess.check_output(
        ['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
    (root / 'robotwin_revision.json').write_text(
        json.dumps(
            {
                'repository':
                'https://github.com/RoboTwin-Platform/RoboTwin.git',
                'branch': 'RLinf_support',
                'commit': commit
            },
            indent=2) + '\n')
    print('CHECKOUT_READY', target, commit, flush=True)


def assets(root):
    import zipfile
    endpoint = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
    repo = 'TianxingChen/RoboTwin2.0'
    dest = root / 'assets'
    info = json.loads(
        fetch(f'{endpoint}/api/datasets/{repo}', dest / 'repo.json'))
    revision = info['sha']
    tree = json.loads(
        fetch(f'{endpoint}/api/datasets/{repo}/tree/{revision}?limit=100',
              dest / 'tree.json'))
    for item in tree:
        if item['path'] not in ('background_texture.zip', 'embodiments.zip',
                                'objects.zip'):
            continue
        target = dest / item['path']
        sha = item['lfs']['oid']
        if not target.exists():
            asset_path = item['path']
            url = (f'{endpoint}/datasets/{repo}/resolve/{revision}/'
                   f'{asset_path}')
            fetch_large(url, target, item['size'], sha)
        with target.open('rb') as stream:
            if digest(stream) != sha:
                raise ValueError(f'Invalid asset checksum: {target}')
        marker = dest / (item['path'] + '.extracted')
        if not marker.exists():
            with zipfile.ZipFile(target) as archive:
                for member in archive.infolist():
                    if not (dest / member.filename).resolve().is_relative_to(
                            dest.resolve()):
                        raise ValueError('Unsafe zip path')
                archive.extractall(dest)
            marker.write_text(sha + '\n')
    (dest / 'manifest.json').write_text(
        json.dumps({
            'repository': repo,
            'revision': revision
        }, indent=2) + '\n')
    # RoboTwin also imports checkout-relative assets via _GLOBAL_CONFIGS.
    # Only create missing links in this experiment's fresh checkout.
    checkout_assets = root / 'src/RoboTwin/assets'
    if checkout_assets.is_dir():
        for name in ('objects', 'embodiments', 'background_texture'):
            link = checkout_assets / name
            if not link.exists() and not link.is_symlink():
                link.symlink_to(dest / name, target_is_directory=True)
            elif link.resolve() != (dest / name).resolve():
                raise ValueError(
                    f'Conflicting experiment asset location: {link}')
    print('ASSETS_READY', dest, flush=True)


def protocol(root):
    import random
    rlinf_root = os.environ.get('RLINF_ROOT')
    if not rlinf_root:
        raise RuntimeError('Set RLINF_ROOT to the RLinf source checkout')
    backend = Path(rlinf_root) / 'rlinf/envs/robotwin/seeds'
    task = 'adjust_bottle'
    train = json.loads(
        (backend / 'train_seeds.json').read_text())[task]['success_seeds']
    test = json.loads(
        (backend / 'eval_seeds.json').read_text())[task]['success_seeds']
    assert len(train) == len(set(train)) == 1000
    assert len(test) == len(set(test)) == 150
    assert not set(train) & set(test)
    validation = random.Random(1234).sample(train, 32)
    training = [seed for seed in train if seed not in set(validation)]
    folder = root / 'protocol'
    folder.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, seeds in [('train', training), ('validation', validation),
                        ('test', test)]:
        content = json.dumps(
            {task: {
                'task_name': task,
                'success_seeds': seeds
            }}, indent=2) + '\n'
        target = folder / f'{name}_seeds.json'
        if target.exists() and target.read_text() != content:
            raise ValueError(f'Refusing to replace existing split: {target}')
        target.write_text(content)
        records[name] = {
            'count': len(seeds),
            'sha256': hashlib.sha256(content.encode()).hexdigest()
        }
    (folder / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    print('PROTOCOL_READY', records, flush=True)


def metadata(root):
    endpoint = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
    dest = root / 'weights' / REPO.split('/')[-1]
    info = json.loads((dest / 'repo.json').read_text())
    revision = info['sha']
    for name in ('config.json', 'model.safetensors.index.json',
                 'physical-intelligence/robotwin/norm_stats.json'):
        fetch(f'{endpoint}/{REPO}/resolve/{revision}/{name}', dest / name)
    tokenizer(dest)
    print('METADATA_READY', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'stage',
        choices=['weights', 'checkout', 'assets', 'protocol', 'metadata'])
    parser.add_argument(
        '--root',
        type=Path,
        default=Path(os.environ.get('FLUX_RL_ROOT', 'work_dirs/rlinf')))
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    globals()[args.stage](args.root)

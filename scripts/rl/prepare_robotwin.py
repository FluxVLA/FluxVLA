#!/usr/bin/env python3
"""Prepare RoboTwin weights, assets and deterministic train/eval splits."""

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


def tokenizer(dest):
    target = dest / 'paligemma_tokenizer.model'
    expected = ('8986bb4f423f07f8c7f70d0dbe3526fb23'
                '16056c17bae71b1ea975e77a168fc6')
    if target.exists() and hashlib.sha256(
            target.read_bytes()).hexdigest() == expected:
        return
    # The public byte mirror is verified against the original tokenizer hash.
    endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co')
    source = (f'{endpoint}/leo009/paligemma_tokenizer.model/'
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
                'Public mirror pinned by content hash'
            },
            indent=2) + '\n')


def weights(root):
    endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co')
    dest = root / 'weights' / REPO.split('/')[-1]
    info = json.loads(
        fetch(f'{endpoint}/api/models/{REPO}', dest / 'repo.json'))
    revision = info['sha']
    tree = json.loads(
        fetch((f'{endpoint}/api/models/{REPO}/tree/{revision}'
               '?recursive=true&limit=100'), dest / 'tree.json'))
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
            subprocess.run([
                'bash',
                str(Path(__file__).with_name('download_hf_ranges.sh')), url,
                str(target),
                str(item['size']), sha, '4'
            ],
                           check=True)
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
    endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co')
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
            asset_name = item['path']
            subprocess.run([
                'bash',
                str(Path(__file__).with_name('download_hf_ranges.sh')),
                f'{endpoint}/datasets/{repo}/resolve/{revision}/{asset_name}',
                str(target),
                str(item['size']), sha, '4'
            ],
                           check=True)
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
    backend_root = Path(
        os.environ.get('RLINF_ROOT',
                       Path(__file__).resolve().parents[3] / 'RLinf'))
    backend = backend_root / 'rlinf/envs/robotwin/seeds'
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
    endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co')
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
        default=Path(
            os.environ.get(
                'FLUX_ROBOTWIN_ROOT',
                Path(__file__).resolve().parents[2] / 'work_dirs/robotwin')))
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    globals()[args.stage](args.root)

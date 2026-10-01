"""Download and verify the pinned Flux SmolVLA LIBERO-10 SFT bundle."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import quote

import requests


def main():
    root = Path(__file__).resolve().parents[2]
    artifacts = Path(
        os.environ.get('SMOLVLA_ARTIFACT_ROOT',
                       root / 'work_dirs/rlinf-smolvla'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output',
        type=Path,
        default=artifacts / 'weights/FluxVLA-SmolVLA-LIBERO10')
    parser.add_argument(
        '--endpoint',
        default=os.environ.get('HF_ENDPOINT', 'https://huggingface.co'))
    args = parser.parse_args()
    repo = 'limxdynamics/FluxVLAEngine'
    revision = '04b5d94d6aebf23ce15b89443df08501740f2868'
    folder = 'smolvla_libero_10_full_finetune_bs64'
    endpoint = args.endpoint.rstrip('/')
    destination = args.output.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    response = requests.get(
        f'{endpoint}/api/models/{repo}/tree/{revision}/{folder}',
        params={
            'recursive': 'true',
            'expand': 'false'
        },
        timeout=30)
    response.raise_for_status()
    entries = response.json()
    manifest = dict(repo=repo, revision=revision, endpoint=endpoint, files=[])
    for entry in entries:
        if entry['type'] != 'file' or entry['path'].endswith('.metadata'):
            continue
        relative = Path(entry['path']).relative_to(folder)
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f'{endpoint}/{repo}/resolve/{revision}/' + quote(
            entry['path'], safe='/')
        lfs = entry.get('lfs')
        if lfs:
            expected = lfs['oid']
            if expected != ('03c02d9e2a2aa1c5f279d38d0366d88b47e88c0'
                            'e46ec44277bef6ac242ec796e'):
                raise ValueError('SFT revision differs from recorded baseline')
            if not path.exists():
                subprocess.run([
                    'bash',
                    str(Path(__file__).with_name('download_hf_ranges.sh')),
                    url,
                    str(path),
                    str(entry['size']), expected, '4'
                ],
                               check=True)
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(data)
            if digest.hexdigest() != expected:
                raise ValueError('Checkpoint SHA256 mismatch')
        else:
            if path.exists():
                data = path.read_bytes()
            else:
                response = requests.get(url, timeout=120)
                response.raise_for_status()
                data = response.content
            blob = f'blob {len(data)}\0'.encode() + data
            if hashlib.sha1(blob).hexdigest() != entry['oid']:
                raise ValueError(f'Git blob hash mismatch: {relative}')
            if not path.exists():
                path.write_bytes(data)
            digest = hashlib.sha256(data)
        if path.stat().st_size != entry['size']:
            raise ValueError(f'File size mismatch: {relative}')
        manifest['files'].append(
            dict(
                path=str(relative),
                sha256=digest.hexdigest(),
                size=entry['size']))
        print('Verified:', relative, flush=True)
    (destination / 'download_manifest.json'
     ).write_text(json.dumps(manifest, indent=2) + '\n')
    print('SFT bundle ready:', destination, flush=True)


if __name__ == '__main__':
    main()

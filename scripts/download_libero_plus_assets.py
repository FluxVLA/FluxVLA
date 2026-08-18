#!/usr/bin/env python3
import argparse
import hashlib
import importlib
import importlib.util
import os
import shutil
import zipfile
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ID = 'Sylvest/LIBERO-plus'
REVISION = 'dd2bd61b7d9a6fef1abc52d606e983b41886a149'
ARCHIVE_NAME = 'assets.zip'
ARCHIVE_SHA256 = (
    '96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf')
REQUIRED_PATHS = (
    'articulated_objects',
    'new_objects',
    'scenes',
    'stable_hope_objects',
    'stable_scanned_objects',
    'textures',
    'turbosquid_objects',
    'serving_region.xml',
    'wall.xml',
)


def _is_libero_plus_root(path):
    return (path / 'setup.py').is_file() and (
        path / 'libero_plus' / 'libero' / 'benchmark' /
        'task_classification.json').is_file()


def _find_libero_plus_root(explicit_root=None):
    if explicit_root is not None:
        root = explicit_root.expanduser().resolve()
        if not _is_libero_plus_root(root):
            raise SystemExit(f'Invalid LIBERO-Plus checkout: {root}')
        return root

    for source_checkout in (PROJECT_ROOT / 'src' / 'libero-plus',
                            PROJECT_ROOT / 'src' / 'LIBERO-plus'):
        if _is_libero_plus_root(source_checkout):
            return source_checkout.resolve()

    spec = None
    if importlib.util.find_spec('libero_plus') is not None:
        spec = importlib.util.find_spec('libero_plus.libero')
    if spec is not None and spec.origin is not None:
        root = Path(spec.origin).resolve().parents[2]
        if _is_libero_plus_root(root):
            return root

    raise AssertionError(
        'Cannot find the LIBERO-Plus checkout. Install it with '
        '`bash scripts/update_env.sh --skip-pull` or pass '
        '--libero-plus-root.')


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as input_file:
        for chunk in iter(lambda: input_file.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _download_archive(cache_dir, endpoint):
    try:
        hf_hub_download = importlib.import_module(
            'huggingface_hub').hf_hub_download
    except ImportError as exc:
        raise AssertionError(
            'huggingface_hub is required. Install it with '
            '`bash scripts/update_env.sh --skip-pull`.') from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type='dataset',
            filename=ARCHIVE_NAME,
            revision=REVISION,
            endpoint=endpoint,
            local_dir=str(cache_dir),
        ))
    actual_hash = _sha256(archive)
    if actual_hash != ARCHIVE_SHA256:
        raise SystemExit(f'LIBERO-Plus asset checksum mismatch: {actual_hash} '
                         f'(expected {ARCHIVE_SHA256})')
    return archive


def _archive_prefix(archive):
    marker = 'LIBERO-plus-0/assets/'
    with zipfile.ZipFile(archive) as zip_file:
        prefixes = {
            name[:name.index(marker) + len(marker)]
            for name in zip_file.namelist() if marker in name
        }
    if len(prefixes) != 1:
        raise SystemExit(
            f'Unexpected LIBERO-Plus archive layout: {sorted(prefixes)}')
    return prefixes.pop()


def _extract_assets(archive, assets_dir):
    prefix = _archive_prefix(archive)
    extract_dir = assets_dir.parent / '.libero-plus-assets-extract'
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)

    try:
        with zipfile.ZipFile(archive) as zip_file:
            for member in zip_file.infolist():
                if member.is_dir() or not member.filename.startswith(prefix):
                    continue
                relative = PurePosixPath(member.filename[len(prefix):])
                if not relative.parts or '..' in relative.parts:
                    continue
                target = extract_dir.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zip_file.open(member) as source, target.open('wb') as out:
                    shutil.copyfileobj(source, out)

        validate_assets(extract_dir)
        if assets_dir.is_symlink():
            assets_dir.unlink()
        elif assets_dir.exists():
            shutil.rmtree(assets_dir)
        extract_dir.rename(assets_dir)
    except Exception:
        shutil.rmtree(extract_dir, ignore_errors=True)
        raise


def validate_assets(assets_dir):
    missing = [
        path for path in REQUIRED_PATHS if not (assets_dir / path).exists()
    ]
    if missing:
        raise SystemExit('LIBERO-Plus asset validation failed. Missing:\n' +
                         '\n'.join(f'  - {path}' for path in missing))
    print(f'Validated LIBERO-Plus assets under {assets_dir}')


def _write_libero_config(root, config_dir):
    try:
        import yaml
    except ImportError as exc:
        raise AssertionError(
            'PyYAML is required by LIBERO-Plus. Install it with '
            '`bash scripts/update_env.sh --skip-pull`.') from exc

    benchmark_root = root / 'libero_plus' / 'libero'
    paths = {
        'benchmark_root': benchmark_root,
        'bddl_files': benchmark_root / 'bddl_files',
        'init_states': benchmark_root / 'init_files',
        'datasets': root / 'libero_plus' / 'datasets',
        'assets': benchmark_root / 'assets',
    }
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / 'config.yaml'
    config_path.write_text(
        yaml.safe_dump(
            {key: str(value.resolve())
             for key, value in paths.items()},
            sort_keys=False),
        encoding='utf-8')
    print(f'Wrote LIBERO config: {config_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--libero-plus-root', type=Path)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument(
        '--endpoint',
        default=os.environ.get('HF_ENDPOINT', 'https://huggingface.co'))
    parser.add_argument(
        '--config-dir',
        type=Path,
        default=(Path(os.environ['LIBERO_PLUS_CONFIG_PATH'])
                 if os.environ.get('LIBERO_PLUS_CONFIG_PATH') else None))
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()

    root = _find_libero_plus_root(args.libero_plus_root)
    assets_dir = root / 'libero_plus' / 'libero' / 'assets'
    cache_dir = (
        args.cache_dir.expanduser().resolve()
        if args.cache_dir is not None else root / 'downloads')

    if args.validate_only:
        validate_assets(assets_dir)
    elif args.force or not assets_dir.exists():
        archive = _download_archive(cache_dir, args.endpoint)
        _extract_assets(archive, assets_dir)
    else:
        try:
            validate_assets(assets_dir)
        except SystemExit:
            archive = _download_archive(cache_dir, args.endpoint)
            _extract_assets(archive, assets_dir)

    config_dir = (
        args.config_dir.expanduser().resolve()
        if args.config_dir is not None else root / '.libero')
    _write_libero_config(root, config_dir)


if __name__ == '__main__':
    main()

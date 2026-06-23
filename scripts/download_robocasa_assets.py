#!/usr/bin/env python3
import argparse
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS_DIR = (
    PROJECT_ROOT / 'src' / 'robocasa-gr1-tabletop-tasks' / 'robocasa' /
    'models' / 'assets')

ARCHIVES = (
    ('robocasa/robocasa-assets', 'objaverse.zip'),
    ('robocasa/robocasa-assets', 'textures.zip'),
    ('robocasa/robocasa-assets', 'generative_textures.zip'),
    # Official fixtures miss appliance visuals used by GR1 tabletop tasks.
    ('jianzhang96/robocasa-assets', 'fixtures.zip'),
    ('nvidia/PhysicalAI-DigitalCousin-Assets', 'sketchfab.zip'),
    ('nvidia/PhysicalAI-DigitalCousin-Assets', 'lightwheel.zip'),
)

REQUIRED_PATHS = (
    'fixtures',
    'textures',
    'generative_textures',
    'objects/objaverse',
    'objects/sketchfab',
    'objects/lightwheel',
    'fixtures/toasters/basic_popup_2/visuals/model_0.obj',
)

BOUND_SITE_NAMES = ('bottom_site', 'top_site', 'horizontal_radius_site')


def _float_list(value):
    return [float(item) for item in value.split()]


def _format_floats(values):
    return ' '.join(f'{value:.10g}' for value in values)


def _merge_tree(src, dst):
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        return
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return

    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            _merge_tree(child, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                target.unlink()
            shutil.move(str(child), str(target))
    src.rmdir()


def _download_archives(cache_dir, endpoint):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit('huggingface_hub is required. Install it with: '
                         'python -m pip install huggingface_hub') from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for repo_id, filename in ARCHIVES:
        print(f'Downloading {repo_id}/{filename}')
        path = hf_hub_download(
            repo_id=repo_id,
            repo_type='dataset',
            filename=filename,
            endpoint=endpoint,
            local_dir=str(cache_dir),
        )
        downloaded.append(Path(path))
    return downloaded


def _extract_archives(archives, assets_dir):
    assets_dir.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        print(f'Extracting {archive.name}')
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(assets_dir)

    objects_dir = assets_dir / 'objects'
    objects_dir.mkdir(parents=True, exist_ok=True)
    for name in ('objaverse', 'sketchfab', 'lightwheel'):
        _merge_tree(assets_dir / name, objects_dir / name)


def normalize_objaverse_sites(assets_dir):
    objaverse_dir = assets_dir / 'objects' / 'objaverse'
    if not objaverse_dir.exists():
        raise FileNotFoundError(
            f'Missing objaverse directory: {objaverse_dir}')

    patched = 0
    unchanged = 0
    missing_bbox = 0
    for model_xml in sorted(objaverse_dir.rglob('model.xml')):
        tree = ET.parse(model_xml)
        root = tree.getroot()
        worldbody = root.find('worldbody')
        outer_body = worldbody.find('body') if worldbody is not None else None
        reg_bbox = next(
            (geom
             for geom in root.iter('geom') if geom.get('name') == 'reg_bbox'),
            None,
        )
        if (outer_body is None or reg_bbox is None
                or reg_bbox.get('pos') is None
                or reg_bbox.get('size') is None):
            missing_bbox += 1
            continue

        pos = _float_list(reg_bbox.get('pos'))
        size = _float_list(reg_bbox.get('size'))
        desired = {
            'bottom_site': [pos[0], pos[1], pos[2] - size[2]],
            'top_site': [pos[0], pos[1], pos[2] + size[2]],
            'horizontal_radius_site': [size[0], size[1], pos[2]],
        }

        direct_sites = {
            site.get('name'): site.get('pos')
            for site in outer_body.findall('site')
            if site.get('name') in BOUND_SITE_NAMES
        }
        if all(
                direct_sites.get(name) == _format_floats(value)
                for name, value in desired.items()):
            unchanged += 1
            continue

        parent_map = {
            child: parent
            for parent in root.iter() for child in list(parent)
        }
        for site in list(root.iter('site')):
            if site.get('name') in BOUND_SITE_NAMES:
                parent = parent_map.get(site)
                if parent is not None:
                    parent.remove(site)

        for name in BOUND_SITE_NAMES:
            ET.SubElement(
                outer_body,
                'site',
                {
                    'name': name,
                    'pos': _format_floats(desired[name]),
                    'rgba': '1 1 1 0.5',
                    'size': '0.005',
                },
            )
        tree.write(model_xml, encoding='utf-8', xml_declaration=False)
        patched += 1

    print(
        'Normalized objaverse XML: '
        f'patched={patched} unchanged={unchanged} missing_bbox={missing_bbox}')
    return patched, unchanged, missing_bbox


def validate_assets(assets_dir):
    missing = [
        path for path in REQUIRED_PATHS if not (assets_dir / path).exists()
    ]
    if missing:
        raise SystemExit('RoboCasa asset validation failed. Missing:\n' +
                         '\n'.join(f'  - {path}' for path in missing))
    print(f'Validated RoboCasa assets under {assets_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets-dir', type=Path, default=DEFAULT_ASSETS_DIR)
    parser.add_argument(
        '--cache-dir', type=Path, default=Path('/tmp/robocasa-assets'))
    parser.add_argument('--endpoint', default='https://hf-mirror.com')
    parser.add_argument('--normalize-only', action='store_true')
    args = parser.parse_args()

    assets_dir = args.assets_dir.resolve()
    if not args.normalize_only:
        archives = _download_archives(args.cache_dir.resolve(), args.endpoint)
        _extract_archives(archives, assets_dir)
    normalize_objaverse_sites(assets_dir)
    validate_assets(assets_dir)


if __name__ == '__main__':
    main()

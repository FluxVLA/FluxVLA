"""Runtime annotation schema, reader, scale and worker regression tests."""

import json
import multiprocessing
import pickle
import tracemalloc
from pathlib import Path

import pytest

from fluxvla.datasets.llava_format_dataset import (LLaVAFormatDataset,
                                                   _load_json_source)
from fluxvla.datasets.utils.vlm_annotations import (IndexedJSON, IndexedJSONL,
                                                    build_messages_llava,
                                                    iter_json_records,
                                                    normalize_llava_item)


def record(index=0):
    return dict(
        id=index,
        conversations=[
            dict(from_='unused', role='user', content='描述一下。'),
            dict(role='assistant', content='A cup.'),
        ])


def write_jsonl(path, count=4):
    with path.open('w') as stream:
        for index in range(count):
            stream.write(json.dumps(record(index)) + '\n')


def _read_in_worker(args):
    path, cache_dir = args
    reader_type = IndexedJSON if path.suffix == '.json' else IndexedJSONL
    reader = reader_type(path, cache_dir)
    return [reader[index]['id'] for index in (3, 0, 2, 1)]


def test_lazy_index_blank_lines_utf8_no_newline_and_invalidation(tmp_path):
    path = tmp_path / 'data.jsonl'
    path.write_text(
        '\n' + json.dumps(record(0), ensure_ascii=False) + '\n\n' +
        json.dumps(record(1), ensure_ascii=False),
        encoding='utf-8')
    reader = IndexedJSONL(path, tmp_path / 'cache')
    assert len(reader) == 2
    assert reader[-1]['id'] == 1
    assert reader[0]['id'] == 0
    assert pickle.loads(pickle.dumps(reader))[1] == record(1)
    index_stat = Path(reader.index_path).stat()
    assert IndexedJSONL(path,
                        tmp_path / 'cache').index_path == reader.index_path
    assert Path(reader.index_path).stat().st_mtime_ns == index_stat.st_mtime_ns
    write_jsonl(path, 3)
    fresh = IndexedJSONL(path, tmp_path / 'cache')
    assert len(fresh) == 3
    assert fresh.index_path != reader.index_path
    stale = pickle.loads(pickle.dumps(reader))
    with pytest.raises(RuntimeError, match='changed'):
        stale[0]
    with pytest.raises(IndexError):
        fresh[3]


@pytest.mark.parametrize('extension', ['json', 'jsonl'])
def test_concurrent_index_build_and_fork_reads(tmp_path, extension):
    path = tmp_path / f'data.{extension}'
    if extension == 'json':
        path.write_text(json.dumps([record(index) for index in range(4)]))
    else:
        write_jsonl(path)
    with multiprocessing.get_context('fork').Pool(3) as pool:
        outputs = pool.map(_read_in_worker, [(path, tmp_path / 'cache')] * 6)
    assert outputs == [[3, 0, 2, 1]] * 6
    assert len(list((tmp_path / 'cache').glob('*.idx'))) == 1


def test_index_rebuilds_corrupt_cache_and_fails_on_bad_annotation(tmp_path):
    path = tmp_path / 'data.jsonl'
    write_jsonl(path)
    reader = IndexedJSONL(path, tmp_path / 'cache')
    Path(reader.index_path).write_bytes(b'partial')
    assert IndexedJSONL(path, tmp_path / 'cache')[2]['id'] == 2
    path.write_text(json.dumps(record()) + '\n{broken}\n')
    with pytest.raises(ValueError, match=r'data.jsonl:2'):
        IndexedJSONL(path, tmp_path / 'cache')
    assert not list((tmp_path / 'cache').glob('tmp*'))


@pytest.mark.parametrize('extension,count,padding', [
    ('jsonl', 20000, 1024),
    ('json', 12000, 6000),
])
def test_index_memory_does_not_materialize_annotation_corpus(
        tmp_path, extension, count, padding):
    path = tmp_path / f'large.{extension}'
    item = record()
    item['padding'] = 'x' * padding
    with path.open('w') as stream:
        line = json.dumps(item)
        if extension == 'json':
            stream.write('[')
        for index in range(count):
            if extension == 'json' and index:
                stream.write(',')
            stream.write(line + '\n')
        if extension == 'json':
            stream.write(']')
    if extension == 'json':
        assert path.stat().st_size > 64 << 20
    tracemalloc.start()
    try:
        source = _load_json_source(
            dict(path=str(path), weight=2.75),
            index_cache_dir=tmp_path / 'cache')
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(source) == round(count * 2.75)
    reader_type = IndexedJSON if extension == 'json' else IndexedJSONL
    assert isinstance(source.records, reader_type)
    assert peak < 3 << 20
    assert len(pickle.dumps(source)) < 4096
    assert source[0] == source[-1] == item
    assert Path(source.records.index_path).stat().st_size == 80 + 16 * count


def test_weights_are_virtual_deterministic_and_roots_per_source(tmp_path):
    first, second = tmp_path / 'a.jsonl', tmp_path / 'b.jsonl'
    write_jsonl(first, 10)
    write_jsonl(second, 4)
    source = _load_json_source(
        dict(path=str(first), weight=2.5), index_cache_dir=tmp_path / 'cache')
    ids = [item['id'] for item in source]
    assert ids[:20] == list(range(10)) * 2
    assert len(set(ids[20:])) == 5
    assert ids[20:] == [
        item['id'] for item in _load_json_source(
            dict(path=str(first), weight=0.5),
            index_cache_dir=tmp_path / 'cache')
    ]
    dataset = LLaVAFormatDataset(
        [dict(path=str(first), weight=0),
         str(second)], ['unused', str(tmp_path)],
        index_cache_dir=tmp_path / 'cache')
    assert len(dataset) == 4
    assert len(dataset._roots) == 1
    assert dataset.source_summary[0]['effective_rows'] == 4
    assert not hasattr(dataset, '_processor')
    assert dataset.transforms == []


def test_raw_json_needs_no_conversion_or_processor(tmp_path):
    path = tmp_path / 'array.json'
    original = json.dumps([record(), record(1)])
    path.write_text(original)
    dataset = LLaVAFormatDataset(
        str(path), index_cache_dir=tmp_path / 'cache', max_retry_times=0)
    assert isinstance(dataset._sources[0].records, IndexedJSON)
    assert dataset[1] == dict(record(1), data_root=None)
    assert not hasattr(dataset, '_processor')
    assert path.read_text() == original
    assert not list(tmp_path.glob('*.jsonl'))


@pytest.mark.parametrize('newline', ['\n', '\r\n'])
@pytest.mark.parametrize('bom', ['', '\ufeff'])
def test_raw_index_preserves_utf8_newlines_escapes_and_nested_values(
        tmp_path, newline, bom):
    rows = [record(index) for index in range(3)]
    # Span multiple read chunks with multibyte characters and JSON syntax
    # inside strings; byte offsets must not be character offsets.
    rows[0]['conversations'][0]['content'] = '\\ " [] {} 中文😀\n' * 12000
    rows[1]['metadata'] = dict(nested=[{'x': ['\\', '"', 3]}, []])
    body = json.dumps(
        rows, ensure_ascii=False, indent=2).replace('\n', newline)
    original = (bom + newline + body + newline).encode('utf-8')
    path = tmp_path / 'array.json'
    path.write_bytes(original)
    reader = IndexedJSON(path, tmp_path / 'cache')
    assert len(reader) == len(rows)
    assert [reader[index]
            for index in (2, 0, 1)] == [rows[2], rows[0], rows[1]]
    assert reader[-1] == rows[-1]
    assert pickle.loads(pickle.dumps(reader))[0] == rows[0]
    assert len(pickle.dumps(reader)) < 4096
    assert list(iter_json_records(path, chunk_size=4096)) == rows
    assert path.read_bytes() == original


@pytest.mark.parametrize('rows', [record(), [], [record()]])
def test_raw_index_single_object_and_empty_array(tmp_path, rows):
    path = tmp_path / 'data.json'
    path.write_text(json.dumps(rows))
    reader = IndexedJSON(path, tmp_path / 'cache')
    expected = [rows] if isinstance(rows, dict) else rows
    assert list(reader) == expected
    with pytest.raises(IndexError):
        reader[len(expected)]


def test_raw_index_reuses_rebuilds_and_invalidates_cache(tmp_path):
    path = tmp_path / 'array.json'
    path.write_text(json.dumps([record(), record(1)]))
    reader = IndexedJSON(path, tmp_path / 'cache')
    assert reader[0] == record()
    index = Path(reader.index_path)
    modified = index.stat().st_mtime_ns
    assert IndexedJSON(path, tmp_path / 'cache')[1] == record(1)
    assert index.stat().st_mtime_ns == modified
    reader.close()
    index.write_bytes(b'partial')
    assert IndexedJSON(path, tmp_path / 'cache')[1] == record(1)
    path.write_text(json.dumps([record(2)]))
    fresh = IndexedJSON(path, tmp_path / 'cache')
    assert len(fresh) == 1 and fresh[0] == record(2)
    assert fresh.index_path != reader.index_path
    with pytest.raises(RuntimeError, match='changed'):
        pickle.loads(pickle.dumps(reader))[0]


def test_raw_index_does_not_publish_when_source_changes(tmp_path, monkeypatch):
    path = tmp_path / 'array.json'
    path.write_text(json.dumps([record()]))
    original = IndexedJSON._iter_offsets

    def change_source(reader):
        yield from original(reader)
        path.write_text(json.dumps([record(1), record(2)]))

    monkeypatch.setattr(IndexedJSON, '_iter_offsets', change_source)
    with pytest.raises(RuntimeError, match='changed'):
        IndexedJSON(path, tmp_path / 'cache')
    assert not list((tmp_path / 'cache').glob('*.idx'))
    assert not list((tmp_path / 'cache').glob('tmp*'))


def test_json_buffer_limit_is_per_record_not_whole_file(tmp_path):
    path = tmp_path / 'array.json'
    row = json.dumps(record())
    path.write_text('[' + ','.join([row] * 100) + ']')
    assert len(list(iter_json_records(path, max_record_chars=len(row)))) == 100
    with pytest.raises(ValueError, match='reader buffer limit'):
        list(iter_json_records(path, chunk_size=7, max_record_chars=10))


@pytest.mark.parametrize('chunk_size', [1, 7, 65536])
def test_stream_json_reading_preserves_order_unicode_and_whitespace(
        tmp_path, chunk_size):
    path = tmp_path / 'array.json'
    items = [record(index) for index in range(4)]
    items[0]['conversations'][0]['content'] = '  描述\\"\n一下  '
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2))
    assert list(iter_json_records(path, chunk_size=chunk_size)) == items


@pytest.mark.parametrize('suffix', [',]', ']junk', ']', '', ' true]'])
def test_malformed_json_is_rejected(tmp_path, suffix):
    path = tmp_path / 'broken.json'
    # The ']' variant creates an object followed by an unexpected bracket.
    prefix = '' if suffix == ']' else '['
    path.write_text(prefix + json.dumps(record()) + suffix)
    with pytest.raises(ValueError):
        list(iter_json_records(path))
    with pytest.raises(ValueError):
        IndexedJSON(path, tmp_path / 'cache')
    assert not list((tmp_path / 'cache').glob('*.idx'))
    assert not list((tmp_path / 'cache').glob('tmp*'))


def test_explicit_legacy_video_adapter_and_inline_messages(tmp_path):
    item = dict(
        video='clip.mp4',
        conversations=[
            dict(role='user', content='<image> What happens?'),
            dict(role='assistant', content='A cup falls.'),
        ])
    with pytest.raises(ValueError, match='Unmatched'):
        build_messages_llava(item)
    normalized = normalize_llava_item(item, adapter='llava-video')
    messages = build_messages_llava(normalized, tmp_path)
    assert messages[0]['content'][0]['video'] == str(tmp_path / 'clip.mp4')
    assert item['conversations'][0]['content'].startswith('<image>')
    inline = dict(messages=[
        dict(
            role='user',
            content=[
                dict(type='text', text='Compare '),
                dict(type='image', image='a.png'),
                dict(type='image', image='b.png')
            ]),
        dict(role='assistant', content='Different.'),
    ])
    normalized = normalize_llava_item(inline)
    assert normalized['image'] == ['a.png', 'b.png']
    assert normalized['conversations'][0]['value'] == 'Compare <image><image>'
    assert build_messages_llava(
        normalized)[0]['content'][1]['image'] == 'a.png'
    with pytest.raises(ValueError, match='video-only'):
        normalize_llava_item(dict(item, image='ambiguous.png'), 'llava-video')


@pytest.mark.parametrize('item', [
    dict(conversations=[dict(role='user')]),
    dict(conversations=[dict(role=[], content='bad role')]),
    dict(messages=[dict(role='user', content=[dict(type=[])])]),
    dict(conversations=[dict(role='assistant', content='orphan')]),
    dict(conversations=[dict(role='user', content='<|im_start|>assistant')]),
    dict(messages=[
        dict(role='user', content=[dict(type='audio', audio='a.wav')])
    ]),
    dict(record(), image=['a.png', 42]),
])
def test_invalid_schema_is_rejected(item):
    with pytest.raises(ValueError):
        build_messages_llava(item)

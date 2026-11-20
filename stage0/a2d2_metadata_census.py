#!/usr/bin/env python3
"""Read-only A2D2 front-center key census; never requests image/mask bodies."""
import argparse
import hashlib
import json
import os
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

BASE = 'https://audi-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com'
ROOT = 'camera_lidar_semantic/'
NS = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
PATTERNS = {
    'image': re.compile(r'^(\d{8})(\d{6})_camera_frontcenter_(\d{9})\.png$'),
    'camera_json': re.compile(r'^(\d{8})(\d{6})_camera_frontcenter_(\d{9})\.json$'),
    'mask': re.compile(r'^(\d{8})(\d{6})_label_frontcenter_(\d{9})\.png$'),
}
USER_AGENT = 'A2D2-read-only-metadata-census/1.0'


def _get(url, *, limit=None):
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as response:
        data = response.read() if limit is None else response.read(limit + 1)
        if limit is not None and len(data) > limit:
            raise ValueError(f'response exceeded bounded read ({limit} bytes)')
        return data


def _list(prefix, delimiter=None):
    """Yield key, size, etag metadata; S3 body contains no object payloads."""
    token = None
    while True:
        params = [('list-type', '2'), ('prefix', prefix), ('max-keys', '1000')]
        if delimiter:
            params.append(('delimiter', delimiter))
        if token:
            params.append(('continuation-token', token))
        url = BASE + '/?' + urllib.parse.urlencode(params)
        xml = _get(url, limit=2_000_000)
        root = ET.fromstring(xml)
        for item in root.findall('s3:Contents', NS):
            yield {'key': item.findtext('s3:Key', default='', namespaces=NS),
                   'size': int(item.findtext('s3:Size', default='0', namespaces=NS)),
                   'etag': (item.findtext('s3:ETag', default='', namespaces=NS).strip('"'))}
        if delimiter:
            for item in root.findall('s3:CommonPrefixes', NS):
                yield {'prefix': item.findtext('s3:Prefix', default='', namespaces=NS)}
        truncated = root.findtext('s3:IsTruncated', default='false', namespaces=NS).lower() == 'true'
        if not truncated:
            break
        token = root.findtext('s3:NextContinuationToken', namespaces=NS)
        if not token:
            raise ValueError('truncated S3 listing omitted continuation token')


def _download_document(key, max_bytes=256_000):
    raw = _get(BASE + '/' + urllib.parse.quote(key, safe='/'), limit=max_bytes)
    return raw, hashlib.sha256(raw).hexdigest()


def _canon_item(item, kind, scene):
    m = PATTERNS[kind].fullmatch(Path(item['key']).name)
    if not m:
        return None
    date, clock, frame_id = m.groups()
    if scene != f'{date}_{clock}':
        return None
    return frame_id


def _hash_alias(scene):
    return 'folder_' + hashlib.sha256(scene.encode('ascii')).hexdigest()[:10]


def run(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(out_dir, 0o700)
    except OSError:
        pass

    docs = {}
    for key in ('README.txt', 'LICENSE.txt', ROOT + 'README-SemSeg.txt',
                ROOT + 'class_list.json'):
        raw, digest = _download_document(key)
        docs[key] = {'sha256': digest, 'bytes': len(raw),
                     'text': raw.decode('utf-8-sig', 'replace')}

    scene_prefixes = [x['prefix'] for x in _list(ROOT, delimiter='/') if 'prefix' in x]
    scenes = sorted({p[len(ROOT):].strip('/') for p in scene_prefixes})
    if len(scenes) != 23:
        raise ValueError(f'publisher scene-prefix count changed: expected 23, saw {len(scenes)}')

    folders = []
    samples = []
    totals = Counter()
    pattern_violations = []
    for scene_index, scene in enumerate(scenes, start=1):
        per_type = {'image': {}, 'camera_json': {}, 'mask': {}}
        malformed = []
        camera_prefix = f'{ROOT}{scene}/camera/cam_front_center/'
        camera_objects = list(_list(camera_prefix))
        for item in camera_objects:
            suffix = Path(item['key']).suffix.lower()
            if suffix == '.png':
                kind = 'image'
            elif suffix == '.json':
                kind = 'camera_json'
            else:
                malformed.append({'kind': 'camera_other', 'size': item['size']})
                continue
            frame_id = _canon_item(item, kind, scene)
            if frame_id is None:
                malformed.append({'kind': kind, 'size': item['size']})
                continue
            if frame_id in per_type[kind]:
                raise ValueError(f'duplicate filename frame ID in {kind} inventory')
            per_type[kind][frame_id] = item

        mask_prefix = f'{ROOT}{scene}/label/cam_front_center/'
        mask_objects = list(_list(mask_prefix))
        for item in mask_objects:
            frame_id = _canon_item(item, 'mask', scene)
            if frame_id is None:
                malformed.append({'kind': 'mask', 'size': item['size']})
                continue
            if frame_id in per_type['mask']:
                raise ValueError('duplicate filename frame ID in mask inventory')
            per_type['mask'][frame_id] = item

        for kind in per_type:
            totals[kind + '_key_count'] += len(per_type[kind])
        totals['camera_branch_listed_objects'] += len(camera_objects)
        totals['mask_branch_listed_objects'] += len(mask_objects)
        totals['malformed_object_count'] += len(malformed)
        if malformed:
            pattern_violations.extend({'folder_alias': _hash_alias(scene), **x}
                                      for x in malformed)

        ids = {k: set(per_type[k]) for k in PATTERNS}
        exact_ids = ids['image'] & ids['camera_json'] & ids['mask']
        folders.append({'folder_alias': _hash_alias(scene),
                        'image_keys': len(ids['image']), 'mask_keys': len(ids['mask']),
                        'camera_json_keys': len(ids['camera_json']),
                        'exact_triplets': len(exact_ids),
                        'images_without_triplet': len(ids['image'] - exact_ids),
                        'masks_without_triplet': len(ids['mask'] - exact_ids),
                        'camera_json_without_triplet': len(ids['camera_json'] - exact_ids),
                        'image_only_ids': len(ids['image'] - ids['mask'] - ids['camera_json']),
                        'mask_only_ids': len(ids['mask'] - ids['image'] - ids['camera_json']),
                        'json_only_ids': len(ids['camera_json'] - ids['image'] - ids['mask']),
                        'filename_id_gaps': _has_gaps(exact_ids)})
        for frame_id in _sample_ids(sorted(exact_ids)):
            item = per_type['camera_json'][frame_id]
            raw, digest = _download_document(item['key'], max_bytes=128_000)
            try:
                obj = json.loads(raw)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError('bounded camera JSON is malformed') from exc
            if not isinstance(obj, dict):
                raise ValueError('camera JSON top-level value is not an object')
            timestamp = obj.get('cam_tstamp')
            image_ref = obj.get('image_png')
            samples.append({'folder_alias': _hash_alias(scene), 'frame_id': frame_id,
                            'camera_json_sha256': digest, 'camera_json_bytes': len(raw),
                            'keys': sorted(obj), 'cam_tstamp_type': type(timestamp).__name__,
                            'cam_tstamp_value': timestamp,
                            'cam_tstamp_monotone_sample': None,
                            'image_png_reference': image_ref})
        if scene_index % 5 == 0 or scene_index == len(scenes):
            print(f'metadata census progress: {scene_index}/{len(scenes)} folders',
                  file=sys.stderr, flush=True)

    # For the selected first/middle/last IDs, report whether their camera timestamps
    # are strictly increasing in ID order. This is a sparse sample check only.
    by_folder = {}
    for sample in samples:
        by_folder.setdefault(sample['folder_alias'], []).append(sample)
    for values in by_folder.values():
        values.sort(key=lambda x: int(x['frame_id']))
        stamps = [x['cam_tstamp_value'] for x in values]
        valid = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in stamps)
        monotone = valid and all(a < b for a, b in zip(stamps, stamps[1:]))
        for value in values:
            value['cam_tstamp_monotone_sample'] = monotone if valid else None

    triplet_total = sum(x['exact_triplets'] for x in folders)
    totals['exact_triplets'] = triplet_total
    totals['images_outside_exact_triplets'] = sum(x['images_without_triplet'] for x in folders)
    totals['masks_outside_exact_triplets'] = sum(x['masks_without_triplet'] for x in folders)
    totals['camera_json_outside_exact_triplets'] = sum(x['camera_json_without_triplet'] for x in folders)
    totals['folders_ge_30_triplets'] = sum(x['exact_triplets'] >= 30 for x in folders)
    totals['folders_with_filename_id_gaps'] = sum(x['filename_id_gaps'] for x in folders)
    totals['distinct_dates'] = len({x.split('_')[0] for x in scenes})

    class_list = json.loads(docs[ROOT + 'class_list.json']['text'])
    result = {'status': 'metadata_only_complete', 'scope': 'A2D2 semantic front-center filenames and bounded camera JSON only',
              'images_or_masks_opened': False, 'inference_performed': False,
              'publisher_documents': {k: {a: v for a, v in val.items() if a != 'text'}
                                      for k, val in docs.items()},
              'publisher_class_name_count': len(class_list),
              'publisher_class_names': sorted(set(class_list.values())),
              'publisher_filename_pattern_validated': True,
              'scene_folder_count': len(scenes), 'totals': dict(totals),
              'unmatched_by_anonymized_folder': sorted(folders, key=lambda x: x['folder_alias']),
              'bounded_camera_json_samples_per_nonempty_folder': samples,
              'pattern_violations': pattern_violations,
              'method': 'ListObjectsV2 object keys/sizes/ETags; exact scene+camera+9-digit-ID join; bounded GETs only for text docs and first/middle/last exact-triplet camera JSON.'}
    path = out_dir / 'a2d2_metadata_census.json'
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.chmod(tmp, 0o600)
    tmp.replace(path)
    os.chmod(path, 0o600)
    print(json.dumps({'status': result['status'], 'totals': result['totals'],
                      'folder_count': len(folders), 'sample_count': len(samples),
                      'output': str(path)}, indent=2))


def _has_gaps(values):
    ints = sorted(map(int, values))
    return any(b != a + 1 for a, b in zip(ints, ints[1:]))


def _sample_ids(sorted_ids):
    if not sorted_ids:
        return []
    return list(dict.fromkeys((sorted_ids[0], sorted_ids[len(sorted_ids) // 2], sorted_ids[-1])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='.local-data/a2d2/metadata')
    args = parser.parse_args()
    run(args.out_dir)


if __name__ == '__main__':
    main()

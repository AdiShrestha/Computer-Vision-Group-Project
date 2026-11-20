#!/usr/bin/env python3
"""Locked 84-frame A2D2 static transfer check; selected RGB/mask objects only."""
import argparse
import hashlib
import io
import json
import math
import os
import platform
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.confseg_core.vkitti import frame_score, restore_native_probabilities, set_size_statistics
from stage0.vkitti_score import EXPECTED_CHECKPOINT, _load_model
from stage0.vkitti_train import CLASSES, IGNORE, preprocess, setup_determinism, sha256

BASE = 'https://audi-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com'
PREFIX = 'camera_lidar_semantic/'
CLASS_KEY = PREFIX + 'class_list.json'
CLASS_SHA = '985606925addb513e3d4164e1bb2c6b2b46f1bb23dc44f8fcb922d8af99ad6d1'
CENSUS_SHA = '7543b5a4b916b6b9a8a91d4e29828e095048184bd328432a5e0af44b7049da7e'
THRESHOLD = 0.9931015335023403
EXPECTED_SCENES = 23
EXPECTED_TRIPLETS = 28652
SELECT_PER_FOLDER = 4
NS = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
MAP_NAMES = {
    'Buildings': 'Building',
    'RD normal street': 'Road',
    'Traffic sign 1': 'TrafficSign', 'Traffic sign 2': 'TrafficSign', 'Traffic sign 3': 'TrafficSign',
    'Traffic signal 1': 'TrafficLight', 'Traffic signal 2': 'TrafficLight', 'Traffic signal 3': 'TrafficLight',
    'Poles': 'Pole', 'Truck 1': 'Truck', 'Truck 2': 'Truck', 'Truck 3': 'Truck',
    'Car 1': 'Car', 'Car 2': 'Car', 'Car 3': 'Car', 'Car 4': 'Car', 'Sky': 'Sky',
}
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASSES)}
MAP_INDEX = {name: CLASS_TO_INDEX[target] for name, target in MAP_NAMES.items()}
USER_AGENT = 'A2D2-static-transfer-locked-sample/1.0'


class TransferError(RuntimeError):
    pass


def _get(url, *, headers=None, limit=None):
    request_headers = {'User-Agent': USER_AGENT}
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(url, headers=request_headers)
    with urllib.request.urlopen(request, timeout=90) as response:
        body = response.read() if limit is None else response.read(limit + 1)
        if limit is not None and len(body) > limit:
            raise TransferError('bounded response size exceeded')
        return body, {k.lower(): v for k, v in response.headers.items()}


def _list(prefix, delimiter=None):
    token = None
    while True:
        params = [('list-type', '2'), ('prefix', prefix), ('max-keys', '1000')]
        if delimiter:
            params.append(('delimiter', delimiter))
        if token:
            params.append(('continuation-token', token))
        body, _ = _get(BASE + '/?' + urllib.parse.urlencode(params), limit=2_000_000)
        try:
            root = ET.fromstring(body)
        except ET.ParseError as exc:
            raise TransferError('S3 listing returned invalid XML') from exc
        for item in root.findall('s3:Contents', NS):
            yield {'key': item.findtext('s3:Key', default='', namespaces=NS),
                   'size': int(item.findtext('s3:Size', default='0', namespaces=NS)),
                   'etag': item.findtext('s3:ETag', default='', namespaces=NS).strip('"')}
        if delimiter:
            for item in root.findall('s3:CommonPrefixes', NS):
                yield {'prefix': item.findtext('s3:Prefix', default='', namespaces=NS)}
        truncated = root.findtext('s3:IsTruncated', default='false', namespaces=NS).lower() == 'true'
        if not truncated:
            return
        token = root.findtext('s3:NextContinuationToken', namespaces=NS)
        if not token:
            raise TransferError('truncated S3 listing omitted continuation token')


def _read_class_list(path):
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != CLASS_SHA:
        raise TransferError('publisher class-list hash differs from retained census')
    data = json.loads(raw.decode('utf-8-sig'))
    if not isinstance(data, dict) or len(data) != 55:
        raise TransferError('publisher class-list shape/count differs from the retained census')
    palette, names = {}, {}
    for color_hex, name in data.items():
        if not isinstance(color_hex, str) or not isinstance(name, str):
            raise TransferError('publisher class list contains non-string key/value')
        try:
            color = tuple(bytes.fromhex(color_hex.lstrip('#')))
        except ValueError as exc:
            raise TransferError('publisher palette contains an invalid hexadecimal color') from exc
        if len(color) != 3 or len(color_hex.lstrip('#')) != 6 or color in palette:
            raise TransferError('publisher palette has malformed or duplicate RGB colors')
        palette[color] = name
        names[name] = color
    if len(names) != 55 or set(MAP_NAMES) - set(names):
        raise TransferError('publisher palette lacks a required exact mapped class name')
    return raw, digest, palette, names


def _decode_mask(mask_rgb, names):
    """Decode a publisher RGB palette mask; recognized unmapped classes become IGNORE."""
    mask = np.asarray(mask_rgb)
    if mask.ndim != 3 or mask.shape[2] != 3 or mask.dtype != np.uint8:
        raise TransferError('mask must decode as HxWx3 uint8 RGB')
    color_to_name = {tuple(color): name for name, color in names.items()}
    labels = np.full(mask.shape[:2], IGNORE, dtype=np.uint8)
    known = np.zeros(mask.shape[:2], dtype=bool)
    for color, name in color_to_name.items():
        pixels = np.all(mask == color, axis=2)
        known |= pixels
        if name in MAP_INDEX:
            labels[pixels] = MAP_INDEX[name]
    if not known.all():
        unknown = int((~known).sum())
        raise TransferError(f'mask contains {unknown} RGB value(s) outside the publisher palette')
    return labels


def _check_native_dimensions(rgb, labels):
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise TransferError('RGB image must decode as HxWx3 uint8')
    if rgb.shape[:2] != labels.shape:
        raise TransferError('native RGB and mask dimensions disagree')


def _sample_digest(scene, frame_id):
    message = ('a2d2-static-v1\0' + scene + '\0' + frame_id).encode('utf-8')
    return hashlib.sha256(message).hexdigest()


def _select(scene_ids):
    selected = []
    for scene, ids in sorted(scene_ids.items()):
        ranked = sorted((_sample_digest(scene, frame), int(frame), frame) for frame in ids)
        if len(ranked) < SELECT_PER_FOLDER:
            raise TransferError(f'{scene} has fewer than four exact triplets')
        selected.extend((scene, frame, digest) for digest, _, frame in ranked[:SELECT_PER_FOLDER])
    return selected


def self_test(class_list_path, out_path):
    raw, class_hash, _, names = _read_class_list(class_list_path)
    colors = {name: tuple(bytes.fromhex(hex_value.lstrip('#'))) for hex_value, name in json.loads(raw).items()}
    # Three mapped classes plus an explicitly excluded, recognized source class.
    fixture_names = ('Buildings', 'RD normal street', 'Traffic sign 1', 'Non-drivable street')
    fixture = np.asarray([[colors[n] for n in fixture_names[:3]] + [colors[fixture_names[3]]]], dtype=np.uint8)
    decoded = _decode_mask(fixture, names)
    expected = np.asarray([[CLASS_TO_INDEX['Building'], CLASS_TO_INDEX['Road'],
                            CLASS_TO_INDEX['TrafficSign'], IGNORE]], dtype=np.uint8)
    if not np.array_equal(decoded, expected):
        raise TransferError('synthetic mapped/excluded palette fixture failed')
    _check_native_dimensions(np.zeros((1, 4, 3), dtype=np.uint8), decoded)
    varied_shape = np.tile(np.asarray([colors['Buildings']], dtype=np.uint8), (3, 5, 1))
    varied_labels = _decode_mask(varied_shape, names)
    _check_native_dimensions(np.zeros((3, 5, 3), dtype=np.uint8), varied_labels)
    if varied_labels.shape != (3, 5) or np.any(varied_labels != CLASS_TO_INDEX['Building']):
        raise TransferError('synthetic varied-native-dimension palette fixture failed')
    try:
        _check_native_dimensions(np.zeros((2, 4, 3), dtype=np.uint8), decoded)
    except TransferError:
        pass
    else:
        raise TransferError('synthetic RGB/mask dimension mismatch was not rejected')
    try:
        _decode_mask(np.asarray([[[1, 2, 3]]], dtype=np.uint8), names)
    except TransferError:
        pass
    else:
        raise TransferError('synthetic unknown palette color was not rejected')
    all_ignored = _decode_mask(np.asarray([[colors['Non-drivable street']]], dtype=np.uint8), names)
    if int(np.sum(all_ignored != IGNORE)) != 0:
        raise TransferError('synthetic all-ignored palette fixture failed')
    # Selection is deterministic and keyed by the full scene name and padded ID.
    fixture_ids = {'camera_lidar_semantic/scene_A': ['000000002', '000000001', '000000003', '000000004', '000000005']}
    first, second = _select(fixture_ids), _select(fixture_ids)
    if first != second or len(first) != 4 or any(len(row[1]) != 9 for row in first):
        raise TransferError('synthetic sample-selection reproducibility fixture failed')
    receipt = json.loads(Path(ROOT / '.local-data/vkitti2/calibration/calibration_receipt.json').read_text())
    if receipt.get('calibration_anchor') != THRESHOLD or receipt.get('status') != 'complete':
        raise TransferError('frozen Scene18 calibration anchor fixture failed')
    # This static path has no target-fitting operation: its threshold input is a constant.
    target_scores = np.asarray([0., 1., .2])
    fixed = THRESHOLD
    target_scores[:] = 0.5
    if fixed != THRESHOLD:
        raise TransferError('synthetic no-target-calibration invariant failed')
    result = {'status': 'passed', 'check': 'synthetic_palette_sample_and_fixed_threshold',
              'publisher_class_list_sha256': class_hash, 'palette_entries': len(names),
              'mapped_classes': len(MAP_NAMES), 'synthetic_palette_pixels': int(fixture.shape[0] * fixture.shape[1]),
              'mapped_fixture_labels': decoded.tolist(), 'excluded_category_label': int(decoded[0, 3]),
              'varied_native_shape_fixture': list(varied_labels.shape),
              'all_ignored_fixture_valid_pixels': int(np.sum(all_ignored != IGNORE)),
              'dimension_mismatch_rejected': True, 'unknown_palette_color_rejected': True,
              'selection_reproducible': True,
              'target_calibration_performed': False, 'fixed_scene18_threshold': THRESHOLD,
              'target_score_mutation_threshold_unchanged': fixed == THRESHOLD}
    _write_private(out_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def _write_private(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + '\n')
    os.chmod(temp, 0o600)
    temp.replace(path)
    os.chmod(path, 0o600)


def lock_manifest(class_list_path, census_path, out_path):
    _, class_hash, _, _ = _read_class_list(class_list_path)
    if class_hash != json.loads(Path(census_path).read_text())['publisher_documents'][CLASS_KEY]['sha256']:
        raise TransferError('class-list digest differs from retained A2D2 metadata census')
    folders = [x['prefix'].rstrip('/') for x in _list(PREFIX, delimiter='/') if 'prefix' in x]
    scenes = sorted(set(folders))
    if len(scenes) != EXPECTED_SCENES:
        raise TransferError(f'expected 23 scene folders, observed {len(scenes)}')
    triplets_by_scene = {}
    scene_ids = {}
    counts = {'image': 0, 'mask': 0, 'camera_json': 0, 'exact_triplets': 0}
    for scene in scenes:
        branches = {
            'image': scene + '/camera/cam_front_center/',
            'camera_json': scene + '/camera/cam_front_center/',
            'mask': scene + '/label/cam_front_center/',
        }
        objects = {kind: {} for kind in branches}
        expected_prefix = scene.rsplit('/', 1)[-1].replace('_', '') + '_camera_frontcenter_'
        for item in _list(branches['image']):
            name = item['key'].rsplit('/', 1)[-1]
            if name.endswith('.png'):
                kind, suffix = 'image', '.png'
            elif name.endswith('.json'):
                kind, suffix = 'camera_json', '.json'
            else:
                continue
            stem = name[:-len(suffix)]
            if not stem.startswith(expected_prefix):
                continue
            frame = stem[-9:]
            if not frame.isdigit():
                raise TransferError(f'malformed front-center {kind} filename')
            if frame in objects[kind]:
                raise TransferError(f'duplicate {kind} frame ID in {scene}')
            objects[kind][frame] = item
        for item in _list(branches['mask']):
            name = item['key'].rsplit('/', 1)[-1]
            expected_prefix = scene.rsplit('/', 1)[-1].replace('_', '') + '_label_frontcenter_'
            if not name.endswith('.png') or not name.startswith(expected_prefix):
                continue
            frame = name[:-4][-9:]
            if not frame.isdigit():
                raise TransferError(f'malformed front-center mask filename in {scene}')
            if frame in objects['mask']:
                raise TransferError(f'duplicate mask frame ID in {scene}')
            objects['mask'][frame] = item
        for kind in counts:
            if kind != 'exact_triplets':
                counts[kind] += len(objects[kind])
        exact = set(objects['image']) & set(objects['camera_json']) & set(objects['mask'])
        counts['exact_triplets'] += len(exact)
        triplets_by_scene[scene] = {frame: {kind: objects[kind][frame] for kind in objects}
                                    for frame in exact}
        if exact:
            scene_ids[scene] = sorted(exact)
    if (len(scene_ids) != 21 or counts['exact_triplets'] != EXPECTED_TRIPLETS or
            counts['image'] != 31448 or counts['mask'] != 31448 or counts['camera_json'] != 28652):
        raise TransferError(f'current key inventory differs from accepted census: {counts}, nonempty={len(scene_ids)}')
    selected = _select(scene_ids)
    rows = []
    for scene, frame, digest in selected:
        item = triplets_by_scene[scene][frame]
        rows.append({'scene_key': scene, 'scene_folder': scene.rsplit('/', 1)[-1],
                     'date': scene.rsplit('/', 1)[-1][:8], 'frame_id': frame,
                     'selection_sha256': digest,
                     'objects': {kind: {'key': entry['key'], 'length': entry['size'], 'etag': entry['etag']}
                                 for kind, entry in sorted(item.items())}})
    if len(rows) != 84:
        raise TransferError(f'expected locked sample of 84 rows, got {len(rows)}')
    body = {'status': 'locked_before_pixel_access', 'protocol': 'a2d2-static-v1',
            'selection_rule': 'lowest four SHA-256 digests of UTF-8 a2d2-static-v1 NUL scene-key NUL zero-padded-frame-ID per nonempty folder; numeric ID tie-breaker',
            'publisher_class_list_sha256': class_hash, 'retained_census_sha256': CENSUS_SHA,
            'inventory_counts': counts, 'scene_folder_count': len(scenes),
            'nonempty_folder_count': len(scene_ids), 'sample_per_nonempty_folder': SELECT_PER_FOLDER,
            'sample_count': len(rows), 'rows': rows}
    canonical = json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    body['manifest_sha256'] = hashlib.sha256(canonical).hexdigest()
    _write_private(out_path, body)
    print(json.dumps({'status': body['status'], 'inventory_counts': counts,
                      'scene_folder_count': len(scenes), 'nonempty_folder_count': len(scene_ids),
                      'sample_count': len(rows), 'manifest_sha256': body['manifest_sha256'],
                      'manifest': str(out_path)}, indent=2, sort_keys=True))


def _verify_manifest(manifest):
    expected = manifest.get('manifest_sha256')
    body = dict(manifest)
    body.pop('manifest_sha256', None)
    actual = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if expected != actual or manifest.get('status') != 'locked_before_pixel_access':
        raise TransferError('locked manifest digest/status is invalid')
    if len(manifest.get('rows', [])) != 84:
        raise TransferError('locked manifest row count is not 84')


def _download_locked(item, root):
    url = BASE + '/' + urllib.parse.quote(item['key'], safe='/')
    body, headers = _get(url)
    actual_len = int(headers.get('content-length', '-1'))
    actual_etag = headers.get('etag', '').strip('"')
    if actual_len != item['length'] or actual_etag != item['etag'] or len(body) != item['length']:
        raise TransferError(f'object snapshot changed at download: {item["key"]}')
    digest = hashlib.sha256(body).hexdigest()
    local_path = Path(root) / digest[:2] / (digest + Path(item['key']).suffix.lower())
    local_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not local_path.exists():
        local_path.write_bytes(body)
        os.chmod(local_path, 0o600)
    elif sha256(local_path) != digest:
        raise TransferError('content-addressed local object has conflicting bytes')
    return local_path, {'key': item['key'], 'length': actual_len, 'etag': actual_etag,
                        'sha256': digest, 'local_content_address': str(local_path.relative_to(root))}


def _decode_rgb(raw):
    try:
        with Image.open(io.BytesIO(raw)) as image:
            return np.asarray(image.convert('RGB'), dtype=np.uint8).copy()
    except (OSError, ValueError) as exc:
        raise TransferError('selected RGB object cannot be decoded') from exc


def _decode_mask_bytes(raw, names):
    try:
        with Image.open(io.BytesIO(raw)) as image:
            source = image.convert('RGB')
            return np.asarray(source, dtype=np.uint8).copy()
    except (OSError, ValueError) as exc:
        raise TransferError('selected semantic mask cannot be decoded as RGB') from exc


def _score_frame(probabilities, labels):
    valid = labels != IGNORE
    count = int(valid.sum())
    if count == 0:
        return {'scorable': False, 'reason': 'zero_mapped_pixels', 'mapped_pixel_count': 0}
    score_row = frame_score(probabilities, labels)
    top1 = np.argmax(probabilities, axis=0)
    errors = valid & (top1 != labels)
    sizes = ((1. - probabilities.astype(np.float64)) <= THRESHOLD).sum(axis=0)
    empty = sizes == 0
    if empty.any():
        sizes[empty] = 1  # fixed top-1 fallback, preserving the full 14-class competition
    mapped_sizes = sizes[valid]
    counts = np.bincount(labels[valid].astype(np.int64), minlength=len(CLASSES))
    error_counts = np.bincount(labels[errors].astype(np.int64), minlength=len(CLASSES))
    return {'scorable': True, 'mapped_pixel_count': count,
            'native_pixel_count': int(labels.size), 'mapped_pixel_fraction': count / labels.size,
            'score_tau05': score_row['score_tau05'],
            'breach': bool(score_row['score_tau05'] > THRESHOLD),
            'top1_errors': int(errors.sum()), 'top1_error_rate': float(errors.sum() / count),
            'mean_set_size_mapped_pixels': float(mapped_sizes.mean()),
            'per_class_support': {name: int(counts[i]) for i, name in enumerate(CLASSES) if counts[i]},
            'per_class_top1_errors': {name: int(error_counts[i]) for i, name in enumerate(CLASSES) if counts[i]}}


def _summarize(rows, metrics):
    by_folder = defaultdict(list)
    by_date = defaultdict(list)
    for row, metric in zip(rows, metrics):
        if metric['scorable']:
            by_folder[row['scene_folder']].append(metric)
            by_date[row['date']].append(metric)
    def aggregate(values):
        if not values:
            return {'scorable_frames': 0, 'breaches': 0, 'breach_rate': None,
                    'mapped_pixels': 0, 'native_pixels': 0, 'mapped_pixel_fraction': None,
                    'top1_errors': 0, 'top1_error_rate': None, 'mean_set_size_mapped_pixels': None}
        mapped = sum(x['mapped_pixel_count'] for x in values)
        native = sum(x['native_pixel_count'] for x in values)
        errors = sum(x['top1_errors'] for x in values)
        return {'scorable_frames': len(values), 'breaches': sum(x['breach'] for x in values),
                'breach_rate': sum(x['breach'] for x in values) / len(values),
                'mapped_pixels': mapped, 'native_pixels': native,
                'mapped_pixel_fraction': mapped / native if native else None,
                'top1_errors': errors, 'top1_error_rate': errors / mapped if mapped else None,
                'mean_set_size_mapped_pixels': float(np.average(
                    [x['mean_set_size_mapped_pixels'] for x in values],
                    weights=[x['mapped_pixel_count'] for x in values]))}
    folder = {k: aggregate(v) for k, v in sorted(by_folder.items())}
    dates = {k: aggregate(v) for k, v in sorted(by_date.items())}
    equal_date = {key: float(np.mean([value[key] for value in dates.values() if value[key] is not None]))
                  if any(value[key] is not None for value in dates.values()) else None
                  for key in ('breach_rate', 'mapped_pixel_fraction', 'top1_error_rate', 'mean_set_size_mapped_pixels')}
    support = defaultdict(int)
    class_errors = defaultdict(int)
    for metric in metrics:
        for name, value in metric.get('per_class_support', {}).items():
            support[name] += value
        for name, value in metric.get('per_class_top1_errors', {}).items():
            class_errors[name] += value
    return {'overall': aggregate([x for x in metrics if x['scorable']]),
            'by_date_reporting_stratum': dates, 'by_folder': folder,
            'equal_date_descriptive_mean': equal_date,
            'mapped_support_by_vkitti_class': dict(sorted(support.items())),
            'top1_errors_by_vkitti_class': dict(sorted(class_errors.items())),
            'unscorable_frames': [{'scene_folder': row['scene_folder'], 'date': row['date'],
                                    'frame_id': row['frame_id'], 'reason': m['reason']}
                                   for row, m in zip(rows, metrics) if not m['scorable']]}


def run(manifest_path, class_list_path, out_dir):
    started = time.perf_counter()
    manifest = json.loads(Path(manifest_path).read_text())
    _verify_manifest(manifest)
    _, class_hash, _, names = _read_class_list(class_list_path)
    if class_hash != manifest['publisher_class_list_sha256']:
        raise TransferError('class-list bytes changed after manifest lock')
    receipt_path = ROOT / '.local-data/vkitti2/calibration/calibration_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('calibration_anchor') != THRESHOLD or receipt.get('checkpoint', {}).get('weights_sha256') != EXPECTED_CHECKPOINT:
        raise TransferError('frozen Scene18 calibration/checkpoint receipt does not match protocol')
    setup_determinism()
    model, training_record, checkpoint = _load_model(
        ROOT / '.local-data/vkitti2/training/vkitti2_deeplab_mobilenetv3_seed101_epoch6.npz',
        ROOT / '.local-data/vkitti2/training/training_record.json', ROOT / 'stage0/vkitti_train.py',
        ROOT / 'project/vkitti_calibration_replay_protocol.md')
    out_dir = Path(out_dir)
    content_dir = out_dir / 'objects'
    out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(out_dir, 0o700)
    records, metrics, download_records = [], [], []
    for index, row in enumerate(manifest['rows'], 1):
        objects = {}
        bodies = {}
        for kind, identity in row['objects'].items():
            path, record = _download_locked(identity, content_dir)
            bodies[kind] = path.read_bytes()
            objects[kind] = record
        rgb = _decode_rgb(bodies['image'])
        mask_rgb = _decode_mask_bytes(bodies['mask'], names)
        labels = _decode_mask(mask_rgb, names)
        _check_native_dimensions(rgb, labels)
        input_tensor, _, geometry = preprocess(rgb, labels)
        import torch
        with torch.inference_mode():
            logits = model(input_tensor.unsqueeze(0).float().cpu())['out']
        probabilities = restore_native_probabilities(logits, geometry)
        if probabilities.shape[0] != 14:
            raise TransferError('model output class dimension changed')
        metric = _score_frame(probabilities, labels)
        records.append({'scene_folder': row['scene_folder'], 'date': row['date'], 'frame_id': row['frame_id'],
                        'sample_selection_sha256': row['selection_sha256'],
                        'objects': objects, 'native_height': int(labels.shape[0]),
                        'native_width': int(labels.shape[1]), 'preprocessing_geometry': geometry,
                        'metrics': metric})
        metrics.append(metric)
        if not metric['scorable']:
            # Explicitly retain unscorable fixture rows; no replacement.
            continue
        download_records.append({'scene_folder': row['scene_folder'], 'frame_id': row['frame_id'],
                                 'object_sha256': {k: v['sha256'] for k, v in objects.items()}})
        if index % 10 == 0 or index == len(manifest['rows']):
            print(f'static transfer progress: {index}/{len(manifest["rows"])} locked rows', file=sys.stderr, flush=True)
    result = {'status': 'complete' if all(x['scorable'] for x in metrics) else 'complete_with_unscorable_frames',
              'scope': 'fixed static transfer; descriptive sample only',
              'protocol_sha256': sha256(ROOT / 'project/a2d2_static_transfer_protocol.md'),
              'manifest_sha256': manifest['manifest_sha256'], 'class_list_sha256': class_hash,
              'retained_a2d2_census_sha256': CENSUS_SHA,
              'vkitti_census_sha256': sha256(ROOT / '.local-data/vkitti2/census.json'),
              'checkpoint_sha256': checkpoint['weights_sha256'],
              'training_record_sha256': sha256(ROOT / '.local-data/vkitti2/training/training_record.json'),
              'training_code_sha256': checkpoint['training_code_sha256'],
              'calibration_receipt_sha256': sha256(receipt_path),
              'calibration_score_vector_sha256': receipt['score_vector_sha256'],
              'calibration_threshold': THRESHOLD, 'threshold_source': 'fixed Scene18 clone calibration receipt',
              'target_calibration_performed': False, 'target_adaptation_performed': False,
              'causal_replay_performed': False,
              'mapping_source_names': MAP_NAMES, 'model_output_class_indices': MAP_INDEX,
              'all_14_logits_competed': True, 'native_unknown_palette_colors_allowed': False,
              'sample_count': len(records), 'exact_triplet_identity_verification': 'length+ETag checked at GET; SHA-256 recorded',
              'inventory_counts': manifest['inventory_counts'],
              'aggregate_summaries': _summarize(manifest['rows'], metrics),
              'per_frame_records': records,
              'runtime': {'python': platform.python_version(), 'platform': platform.platform(),
                          'numpy': np.__version__, 'pillow': Image.__version__,
                          'torch': __import__('torch').__version__,
                          'torchvision': __import__('torchvision').__version__,
                          'device': 'cpu', 'threads': __import__('torch').get_num_threads(),
                          'deterministic_algorithms': __import__('torch').are_deterministic_algorithms_enabled()},
              'elapsed_seconds': time.perf_counter() - started,
              'limitations': ['dates are reporting strata only, not independent routes',
                              '84 deterministic sampled frames are a descriptive transfer check, not population coverage evidence',
                              'no adaptation, causal feedback, weather effect, or significance test']}
    # Separate private detailed result and public-safe aggregate report locally; no raw imagery leaves object cache.
    private_path = out_dir / 'static_transfer.private.json'
    _write_private(private_path, result)
    aggregate = dict(result)
    aggregate.pop('per_frame_records', None)
    aggregate_path = out_dir / 'static_transfer.aggregate.json'
    _write_private(aggregate_path, aggregate)
    print(json.dumps({'status': result['status'], 'sample_count': len(records),
                      'scorable_frames': result['aggregate_summaries']['overall']['scorable_frames'],
                      'breaches': result['aggregate_summaries']['overall']['breaches'],
                      'mapped_pixels': result['aggregate_summaries']['overall']['mapped_pixels'],
                      'manifest_sha256': manifest['manifest_sha256'],
                      'private_result': str(private_path), 'aggregate_result': str(aggregate_path)},
                     indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    test = sub.add_parser('self-test')
    test.add_argument('--class-list', required=True)
    test.add_argument('--out', required=True)
    lock = sub.add_parser('lock-sample')
    lock.add_argument('--class-list', required=True)
    lock.add_argument('--census', default=str(ROOT / '.local-data/a2d2/metadata/a2d2_metadata_census.json'))
    lock.add_argument('--out', required=True)
    execute = sub.add_parser('run')
    execute.add_argument('--class-list', required=True)
    execute.add_argument('--manifest', required=True)
    execute.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    if args.command == 'self-test':
        self_test(args.class_list, args.out)
    elif args.command == 'lock-sample':
        lock_manifest(args.class_list, args.census, args.out)
    else:
        run(args.manifest, args.class_list, args.out_dir)


if __name__ == '__main__':
    main()

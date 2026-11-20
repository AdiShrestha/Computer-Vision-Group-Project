#!/usr/bin/env python3
"""Single-run, preflight-bound VKITTI 2 Scene02/20 clone/fog/rain comparison."""
import argparse
import copy
import hashlib
import io
import json
import platform
import re
import sys
import time
import traceback
from pathlib import Path
import tarfile
import zipfile

import numpy as np
from PIL import Image
import torch
import torchvision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.confseg_core.vkitti import frame_score, restore_native_probabilities
from core.confseg_core import CausalController  # bound as a used implementation dependency
from replay import vkitti_replay
from replay.vkitti_replay import CONDITIONS, METHODS, PREFIX, SUFFIX, run_scene, summarize_two_scenes
from stage0.vkitti_score import _load_model
from stage0.vkitti_train import CLASSES, IGNORE, _read_tar_bytes, preprocess, setup_determinism

SCENES = ('02', '20')
ADVERSE = ('fog', 'rain')
RGB_RE = re.compile(r'^Scene(02|20)/(clone|fog|rain)/frames/rgb/Camera_0/rgb_(\d+)\.jpg$')
MASK_RE = re.compile(r'^Scene(02|20)/(clone|fog|rain)/frames/classSegmentation/Camera_0/classgt_(\d+)\.png$')
PALETTE_RE = re.compile(r'^Scene(02|20)/clone/colors\.txt$')
EXPECTED_CHECKPOINT = '760e7d1a5e91fea543cd1c6aaa1946d044e23d1e96960c133360d64daf9cab4d'


class HeldoutError(RuntimeError):
    pass


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def _selection_keys():
    keys = []
    for scene in SCENES:
        for condition in CONDITIONS:
            frames = range(60) if condition == 'clone' else range(30, 60)
            keys.extend((scene, condition, frame) for frame in frames)
    return keys


def _index_members(path, pattern, *, kind):
    members = {}
    try:
        with tarfile.open(path, 'r:') as archive:
            for member in archive:
                if not member.isfile():
                    continue
                match = pattern.fullmatch(member.name)
                if not match:
                    continue
                scene, condition, frame_text = match.groups()
                frame = int(frame_text)
                if condition == 'clone':
                    selected = 0 <= frame < 60
                else:
                    selected = 30 <= frame < 60
                if not selected:
                    continue
                key = (scene, condition, frame)
                if key in members:
                    raise HeldoutError(f'duplicate selected {kind} key {key}')
                members[key] = member
    except (OSError, tarfile.TarError) as exc:
        raise HeldoutError(f'cannot scan {kind} TAR headers ({type(exc).__name__})') from exc
    return members


def _selected_indices(rgb_tar_path, seg_tar_path):
    rgb = _index_members(rgb_tar_path, RGB_RE, kind='RGB')
    seg = _index_members(seg_tar_path, MASK_RE, kind='segmentation')
    expected = set(_selection_keys())
    if set(rgb) != expected:
        raise HeldoutError(f'RGB selected key inventory differs: missing={len(expected-set(rgb))}, extra={len(set(rgb)-expected)}')
    if set(seg) != expected:
        raise HeldoutError(f'segmentation selected key inventory differs: missing={len(expected-set(seg))}, extra={len(set(seg)-expected)}')
    return rgb, seg


def _parse_scene_palettes(textgt_path):
    palettes = {}
    try:
        with tarfile.open(textgt_path, 'r:gz') as archive:
            for member in archive:
                if not member.isfile() or not PALETTE_RE.fullmatch(member.name):
                    continue
                scene = member.name[5:7]
                if scene in palettes:
                    raise HeldoutError(f'duplicate publisher palette for scene {scene}')
                rows = [line.split() for line in _read_tar_bytes(archive, member).decode('utf-8-sig').splitlines()
                        if line.strip()]
                if not rows or rows[0] != ['Category', 'r', 'g', 'b']:
                    raise HeldoutError(f'publisher palette header invalid for scene {scene}')
                palette = {}
                for row in rows[1:]:
                    if len(row) != 4:
                        raise HeldoutError('publisher palette row malformed')
                    name = row[0]
                    color = tuple(map(int, row[1:]))
                    if name in palette or any(v < 0 or v > 255 for v in color):
                        raise HeldoutError('publisher palette has duplicate names or invalid values')
                    palette[name] = color
                if len(set(palette.values())) != len(palette):
                    raise HeldoutError('publisher palette contains ambiguous colors')
                palettes[scene] = palette
    except (OSError, tarfile.TarError, UnicodeError, ValueError) as exc:
        if isinstance(exc, HeldoutError):
            raise
        raise HeldoutError(f'cannot read selected publisher palettes ({type(exc).__name__})') from exc
    if set(palettes) != set(SCENES):
        raise HeldoutError('missing a held-out scene palette')
    return palettes


def _synthetic_dry_run(calibration_scores=None):
    """Exercise frame selection/replay/set-size wiring without dataset pixels or writes."""
    selected = set(_selection_keys())
    if len(selected) != 240 or len([k for k in selected if k[1] == 'clone']) != 120:
        raise HeldoutError('frozen frame selection failed synthetic cardinality check')
    missing = selected - {k for k in selected if not (k[1] == 'rain' and k[2] == 59)}
    if len(missing) != 2:
        raise HeldoutError('synthetic frame selector did not detect missing selected keys')
    cal = np.linspace(.01, .99, 339, dtype=np.float64) if calibration_scores is None else np.asarray(calibration_scores, dtype=np.float64)
    clear_scores = np.linspace(.05, .65, 30)
    clear_features = np.tile([.2, .1, .8], (30, 1))
    probs = np.zeros((14, 2, 2), dtype=np.float32)
    probs[0] = .9
    probs[1] = .1
    support = {'guardrail_pixel_count': 1, 'guardrail_top1_error_count': 0}
    suffixes = {}
    for condition, offset in zip(CONDITIONS, (0., .1, .2)):
        suffixes[condition] = {
            'scores': np.clip(np.linspace(.2, .8, 30) + offset, 0, 1),
            'features': np.tile([.3 + offset, .2, .7 - offset], (30, 1)),
            'probability_provider': lambda _i, p=probs: p,
            'class_support_provider': lambda _i, s=support: s,
            'input_artifact_sha256': hashlib.sha256(condition.encode()).hexdigest(),
        }
    one = run_scene('02', cal, clear_scores, clear_features, suffixes)
    two = run_scene('20', cal, clear_scores, clear_features, suffixes)
    summary = summarize_two_scenes([one, two])
    rows = one['arm_results'] + two['arm_results']
    if len(rows) != 42 or len(summary['arm_summaries']) != 21:
        raise HeldoutError('synthetic held-out driver did not emit 42 arms/21 summaries')
    for row in rows:
        if row['first_adverse_feedback_decision_index'] != 31:
            raise HeldoutError('synthetic warm-up feedback binding failed')
        if row['feedback_released_at_switch'] != [{'decision_index': 30, 'frame_index': 29}]:
            raise HeldoutError('synthetic clear-prefix switch release failed')
        m = row['metrics']
        activ2 = float(1. - np.float64(np.float32(.1)))
        expected_size = float(np.mean([2. if t >= activ2 else 1. for t in row['post_switch_thresholds']]))
        expected_multi = float(np.mean([1. if t >= activ2 else 0. for t in row['post_switch_thresholds']]))
        if m['native_mean_set_size'] != expected_size or m['native_fraction_multi_class'] != expected_multi or m['set_size_pixel_count'] != 120:
            raise HeldoutError('synthetic exact float32 set-size callback failed')
        if m['guardrail_pixel_support'] != 30 or m['guardrail_top1_error_count'] != 0:
            raise HeldoutError('synthetic support callback failed')
    return {'status': 'passed', 'dataset_pixels_opened': False,
            'selected_frame_keys_per_scene': 120, 'total_selected_frame_keys': 240,
            'scene_condition_arm_rows': 42, 'two_scene_descriptive_summaries': 21,
            'common_clear_prefix': True, 'switch_feedback': 't=30 receives frame 29; first adverse release at t=31',
            'float32_boundary_callback': 'exact', 'all_arm_set_size_callbacks': 'passed'}


def _load_bindings(root):
    setup_determinism()
    data = root / '.local-data' / 'vkitti2'
    cal_dir = data / 'calibration'
    paths = {
        'rgb': data / 'vkitti_2.0.3_rgb.tar',
        'class_segmentation': data / 'vkitti_2.0.3_classSegmentation.tar',
        'text_metadata': data / 'vkitti_2.0.3_textgt.tar.gz',
        'census': data / 'census.json',
        'weights': data / 'training' / 'vkitti2_deeplab_mobilenetv3_seed101_epoch6.npz',
        'training_record': data / 'training' / 'training_record.json',
        'training_code': root / 'stage0' / 'vkitti_train.py',
        'protocol': root / 'project' / 'vkitti_calibration_replay_protocol.md',
        'original_calibration_receipt': cal_dir / 'calibration_receipt.original.json',
        'current_calibration_receipt': cal_dir / 'calibration_receipt.json',
        'calibration_correction': cal_dir / 'calibration_receipt.correction.json',
        'calibration_scores': cal_dir / 'scene18_clone_scores_tau05.npy',
        'calibration_features': cal_dir / 'scene18_clone_label_free_features.npy',
    }
    missing = [k for k, p in paths.items() if not p.is_file()]
    if missing:
        raise HeldoutError('missing preflight artifacts: ' + ', '.join(missing))
    census = json.loads(paths['census'].read_text())
    training = json.loads(paths['training_record'].read_text())
    current_receipt = json.loads(paths['current_calibration_receipt'].read_text())
    original_bytes = paths['original_calibration_receipt'].read_bytes()
    original = json.loads(original_bytes)
    correction = json.loads(paths['calibration_correction'].read_text())
    if _sha(paths['original_calibration_receipt']) != correction.get('original_receipt_sha256'):
        raise HeldoutError('scope correction does not bind the preserved original calibration receipt')
    if hashlib.sha256(original_bytes).hexdigest() != correction.get('original_receipt_sha256'):
        raise HeldoutError('preserved original calibration receipt hash mismatch')
    if correction.get('accepted_calibration_score_vector_sha256') != original.get('score_vector_sha256'):
        raise HeldoutError('scope correction does not bind the accepted score vector')
    if not correction.get('scope_facts', {}).get('scene18_clone_calibration_pixels_opened'):
        raise HeldoutError('scope correction fails to declare calibration pixels opened')
    if correction.get('scope_facts', {}).get('scene02_pixels_opened_during_calibration') is not False or correction.get('scope_facts', {}).get('scene20_pixels_opened_during_calibration') is not False:
        raise HeldoutError('scope correction does not verify held-out pixels remained closed during calibration')
    if current_receipt.get('score_vector_sha256') != original.get('score_vector_sha256'):
        raise HeldoutError('current/original calibration receipts disagree on accepted vector')
    if current_receipt.get('calibration_pixels_opened') is not True or current_receipt.get('held_out_pixels_opened') is not False:
        raise HeldoutError('current calibration receipt has ambiguous or incorrect scope facts')
    score_bytes_hash = _sha(paths['calibration_scores'])
    if score_bytes_hash != correction.get('calibration_score_file_sha256'):
        raise HeldoutError('calibration score file differs from bound scope correction')
    scores = np.load(paths['calibration_scores'], allow_pickle=False)
    if scores.shape != (339,) or not np.isfinite(scores).all():
        raise HeldoutError('accepted calibration score vector has wrong shape or non-finite values')
    vector_digest = hashlib.sha256(scores.astype('<f8', copy=False).tobytes()).hexdigest()
    if vector_digest != original.get('score_vector_sha256') or vector_digest != correction.get('accepted_calibration_score_vector_sha256'):
        raise HeldoutError('accepted calibration vector digest mismatch')
    anchor = vkitti_replay.calibration_anchor(scores, .10)
    if anchor['rank_one_based'] != 306 or anchor['anchor'] != original.get('calibration_anchor'):
        raise HeldoutError('accepted calibration rank/anchor mismatch')
    if _sha(paths['training_code']) != training.get('training_code_sha256'):
        raise HeldoutError('training source digest no longer matches the fixed checkpoint')
    if _sha(paths['protocol']) != original.get('protocol_sha256') or _sha(paths['protocol']) != current_receipt.get('protocol_sha256'):
        raise HeldoutError('locked protocol digest mismatch')
    if _sha(paths['weights']) != EXPECTED_CHECKPOINT or _sha(paths['weights']) != training.get('weight_file_sha256'):
        raise HeldoutError('final checkpoint digest mismatch')
    expected_archives = {
        'rgb': census['archive_digests']['rgb']['sha256'],
        'class_segmentation': census['archive_digests']['class_segmentation']['sha256'],
        'text_metadata': census['publisher_metadata_sha256'],
    }
    archive_hashes = {k: _sha(paths[k]) for k in expected_archives}
    if archive_hashes != expected_archives:
        raise HeldoutError('publisher archive digest differs from the accepted census')
    for key, digest in expected_archives.items():
        if original['archive_digests'][key]['sha256'] != digest:
            raise HeldoutError(f'calibration receipt source digest differs for {key}')
    palette_expected = training['publisher_palette_rgb']
    palettes = _parse_scene_palettes(paths['text_metadata'])
    for scene in SCENES:
        if {name: list(rgb) for name, rgb in palettes[scene].items()} != palette_expected:
            raise HeldoutError(f'held-out metadata palette differs for scene {scene}')
    selected_rgb, selected_masks = _selected_indices(paths['rgb'], paths['class_segmentation'])
    if set(selected_rgb) != set(selected_masks):
        raise HeldoutError('held-out selected RGB/mask keys do not pair exactly')
    # Strict tensor-only model restore before any held-out member bytes are opened.
    from stage0.vkitti_score import _load_model as load_model
    model, _, checkpoint = load_model(paths['weights'], paths['training_record'],
                                      paths['training_code'], paths['protocol'])
    if checkpoint['weights_sha256'] != EXPECTED_CHECKPOINT:
        raise HeldoutError('strictly loaded model checkpoint digest differs')
    code_paths = {
        'driver': Path(__file__).resolve(),
        'scorer': root / 'stage0' / 'vkitti_score.py',
        'replay': root / 'replay' / 'vkitti_replay.py',
        'controller': root / 'core' / 'confseg_core' / 'controllers.py',
        'score_core': root / 'core' / 'confseg_core' / 'vkitti.py',
        'training': paths['training_code'],
    }
    code_hashes = {name: _sha(path) for name, path in code_paths.items()}
    if code_hashes['scorer'] != original['score_code_sha256']:
        raise HeldoutError('score restoration/scoring source digest differs from calibration')
    dry_run = _synthetic_dry_run(calibration_scores=scores)
    # Recover the exact settings/hashes emitted by the real controller runner using synthetic inputs.
    settings_by_method = {}
    # Each method's setting record is identically bound across all render conditions.
    synthetic_cal = np.asarray(scores, dtype=np.float64)
    fake_clear = np.full(30, .2, dtype=np.float64)
    fake_features = np.tile([.2, .1, .8], (30, 1))
    fake_suffixes = {cond: {'scores': np.full(30, .3), 'features': fake_features}
                     for cond in CONDITIONS}
    synthetic_scene = run_scene('02', synthetic_cal, fake_clear, fake_features, fake_suffixes)
    for row in synthetic_scene['arm_results']:
        settings_by_method[row['method']] = {'settings': row['settings'], 'sha256': row['settings_sha256']}
    runtime = {'python': platform.python_version(), 'torch': torch.__version__,
               'torchvision': torchvision.__version__, 'numpy': np.__version__,
               'pillow': Image.__version__, 'platform': platform.platform(),
               'device': 'cpu', 'torch_threads': torch.get_num_threads(),
               'torch_interop_threads': torch.get_num_interop_threads(),
               'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
               'mps_available': bool(torch.backends.mps.is_available())}
    if runtime['device'] != original['runtime']['device'] or runtime['torch'] != original['runtime']['torch'] or runtime['torchvision'] != original['runtime']['torchvision']:
        raise HeldoutError('held-out runtime identity differs from calibration runtime')
    selections = [{'scene': s, 'condition': c, 'frame': f} for s, c, f in _selection_keys()]
    binding = {'status': 'preflight_passed_no_heldout_pixels_opened',
               'dataset': 'Virtual KITTI 2 v2.0.3', 'selection': selections,
               'selected_frame_count': len(selections), 'scene_count': 2,
               'conditions': list(CONDITIONS), 'methods': list(METHODS),
               'archive_sha256': archive_hashes,
               'archive_bytes': {k: paths[k].stat().st_size for k in expected_archives},
               'checkpoint_sha256': checkpoint['weights_sha256'],
               'training_record_sha256': _sha(paths['training_record']),
               'calibration_original_receipt_sha256': correction['original_receipt_sha256'],
               'calibration_current_receipt_sha256': _sha(paths['current_calibration_receipt']),
               'calibration_scope_correction_sha256': _sha(paths['calibration_correction']),
               'accepted_calibration_score_file_sha256': score_bytes_hash,
               'accepted_calibration_score_vector_sha256': vector_digest,
               'calibration_anchor': anchor['anchor'], 'calibration_rank': anchor['rank_one_based'],
               'protocol_sha256': _sha(paths['protocol']), 'code_sha256': code_hashes,
               'runtime': runtime, 'settings_by_method': settings_by_method,
               'synthetic_dry_run': dry_run,
               'calibration_pixels_opened': True, 'held_out_pixels_opened': False}
    binding['preflight_binding_sha256'] = _canonical_sha(binding)
    return binding, paths, selected_rgb, selected_masks, palettes, model, scores


def preflight(out_dir):
    out_dir = Path(out_dir)
    marker = out_dir / 'run_start.json'
    if marker.exists():
        raise HeldoutError('single-run marker exists; a previous real held-out attempt must be reported before restart')
    binding, _, _, _, _, _, _ = _load_bindings(ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(out_dir / 'preflight_receipt.json', binding)
    return binding


def _decode_pair(rgb_tar, seg_tar, rgb_member, mask_member, color_to_class):
    try:
        with Image.open(io.BytesIO(_read_tar_bytes(rgb_tar, rgb_member))) as src:
            rgb_image = src.convert('RGB')
            rgb = np.asarray(rgb_image, dtype=np.uint8).copy()
            native_size = rgb_image.size
        with Image.open(io.BytesIO(_read_tar_bytes(seg_tar, mask_member))) as src:
            mask = np.asarray(src.convert('RGB'), dtype=np.uint8).copy()
    except (OSError, ValueError) as exc:
        raise HeldoutError('malformed selected held-out RGB/mask pair') from exc
    if native_size != (mask.shape[1], mask.shape[0]):
        raise HeldoutError('selected held-out native RGB/mask dimensions disagree')
    labels = np.full(mask.shape[:2], IGNORE, dtype=np.uint8)
    known = np.zeros(mask.shape[:2], dtype=bool)
    for color, class_id in color_to_class.items():
        match = np.all(mask == color, axis=2)
        labels[match] = class_id
        known |= match
    if not known.all():
        raise HeldoutError(f'selected held-out mask has {int((~known).sum())} unregistered palette pixels')
    return rgb, labels


def _probability_cache_path(out_dir, key):
    scene, condition, frame = key
    return Path(out_dir) / 'probabilities' / f'{scene}_{condition}_{frame:05}.npy'


def _score_and_cache(rgb_path, seg_path, rgb_idx, seg_idx, palettes, model, out_dir, binding):
    out_dir = Path(out_dir)
    prob_dir = out_dir / 'probabilities'
    prob_dir.mkdir(parents=True, exist_ok=True)
    color_maps = {}
    for scene in SCENES:
        palette = palettes[scene]
        color_maps[scene] = {tuple(palette[name]): i for i, name in enumerate(CLASSES)}
        color_maps[scene][tuple(palette['Undefined'])] = IGNORE
    rows = []
    total = len(_selection_keys())
    started = time.perf_counter()
    with tarfile.open(rgb_path, 'r:') as rt, tarfile.open(seg_path, 'r:') as st:
        for i, key in enumerate(_selection_keys(), start=1):
            rgb, labels = _decode_pair(rt, st, rgb_idx[key], seg_idx[key], color_maps[key[0]])
            x, _, geometry = preprocess(rgb, labels)
            with torch.inference_mode():
                logits = model(x.unsqueeze(0).float().cpu())['out']
            probs = restore_native_probabilities(logits, geometry)
            result = frame_score(probs, labels)
            result.pop('pixel_scores')
            prob_path = _probability_cache_path(out_dir, key)
            prob_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(prob_path, probs.astype(np.float32, copy=False), allow_pickle=False)
            row = {'scene': key[0], 'condition': key[1], 'frame_index': key[2],
                   'frame_id': f'Scene{key[0]}/{key[1]}/Camera_0/{key[2]:05}',
                   'score_tau05': result['score_tau05'],
                   'label_free_features': [result['mean_entropy'], result['low_conf_fraction'], result['mean_max_prob']],
                   'valid_pixel_count': result['valid_pixel_count'],
                   'native_class_pixel_counts': result['native_class_pixel_counts'].tolist(),
                   'top1_error_count': result['top1_error_count'],
                   'top1_error_counts_by_class': result['top1_error_counts_by_class'].tolist(),
                   'guardrail_pixel_count': result['guardrail_pixel_count'],
                   'guardrail_top1_error_count': result['guardrail_top1_error_count'],
                   'probability_file': prob_path.relative_to(out_dir).as_posix(),
                   'probability_sha256': _sha(prob_path),
                   'preprocessing_geometry': geometry}
            rows.append(row)
            if i % 10 == 0 or i == total:
                _atomic_json(out_dir / 'progress.json', {'status': 'scoring_in_progress',
                              'processed_selected_frames': i, 'expected_selected_frames': total,
                              'elapsed_seconds': time.perf_counter() - started,
                              'last_key': row['frame_id'], 'calibration_pixels_opened': True,
                              'held_out_pixels_opened': True,
                              'preflight_binding_sha256': binding['preflight_binding_sha256']})
    if len(rows) != total:
        raise HeldoutError(f'held-out scoring produced {len(rows)} rows; expected {total}')
    row_path = out_dir / 'frame_measurements.private.json'
    _atomic_json(row_path, rows)
    return rows, {'frame_measurements_sha256': _sha(row_path),
                  'selected_probability_file_count': total,
                  'selected_probability_bytes': sum(_probability_cache_path(out_dir, k).stat().st_size
                                                   for k in _selection_keys()),
                  'score_inference_elapsed_seconds': time.perf_counter() - started}


def _condition_input_digest(scene, condition, row_lookup):
    h = hashlib.sha256()
    h.update(scene.encode()); h.update(condition.encode())
    keys = [(scene, 'clone', frame) for frame in range(30)] + [(scene, condition, frame) for frame in range(30, 60)]
    for key in keys:
        row = row_lookup[key]
        h.update(row['frame_id'].encode())
        h.update(np.asarray([row['score_tau05'], *row['label_free_features']], dtype='<f8').tobytes())
        h.update(row['probability_sha256'].encode())
        h.update(np.asarray(row['native_class_pixel_counts'], dtype='<i8').tobytes())
        h.update(np.asarray(row['top1_error_counts_by_class'], dtype='<i8').tobytes())
    return h.hexdigest()


def _replay_cached(scene, rows, cal_scores, out_dir):
    row_lookup = {(r['scene'], r['condition'], r['frame_index']): r for r in rows}
    clear_rows = [row_lookup[(scene, 'clone', i)] for i in range(30)]
    clear_scores = [r['score_tau05'] for r in clear_rows]
    clear_features = [r['label_free_features'] for r in clear_rows]
    suffixes = {}
    for condition in CONDITIONS:
        suffix_rows = [row_lookup[(scene, condition, i)] for i in range(30, 60)]
        def prob_provider(frame_index, suffix_rows=suffix_rows):
            row = suffix_rows[frame_index - 30]
            path = Path(out_dir) / row['probability_file']
            return np.load(path, mmap_mode='r', allow_pickle=False)
        def support_provider(frame_index, suffix_rows=suffix_rows):
            row = suffix_rows[frame_index - 30]
            return {'guardrail_pixel_count': row['guardrail_pixel_count'],
                    'guardrail_top1_error_count': row['guardrail_top1_error_count']}
        suffixes[condition] = {'scores': [r['score_tau05'] for r in suffix_rows],
                               'features': [r['label_free_features'] for r in suffix_rows],
                               'probability_provider': prob_provider,
                               'class_support_provider': support_provider,
                               'input_artifact_sha256': _condition_input_digest(scene, condition, row_lookup)}
    return run_scene(scene, cal_scores, clear_scores, clear_features, suffixes)


def run_once(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_marker = out_dir / 'run_start.json'
    if start_marker.exists():
        raise HeldoutError('run_start marker already exists; report the prior attempt and do not restart')
    # Recompute all non-pixel prerequisites and compare with the separately written preflight receipt.
    binding, paths, rgb_idx, seg_idx, palettes, model, cal_scores = _load_bindings(ROOT)
    saved_preflight_path = out_dir / 'preflight_receipt.json'
    if not saved_preflight_path.is_file():
        raise HeldoutError('missing verified held-out preflight receipt')
    saved_preflight = json.loads(saved_preflight_path.read_text())
    if saved_preflight.get('preflight_binding_sha256') != binding['preflight_binding_sha256']:
        raise HeldoutError('preflight bindings changed since synthetic dry run; held-out pixels remain closed')
    # Validate each cached map once during replay setup; do not rehash the same map
    # for every controller callback (there are 1,260 callbacks per scene).
    # Keep archive readers open; indexing below touches only TAR headers, before the start marker.
    rgb_tar = tarfile.open(paths['rgb'], 'r:')
    seg_tar = tarfile.open(paths['class_segmentation'], 'r:')
    try:
        # Reconfirm exact selections from open header inventories without decoding any payload.
        expected = set(_selection_keys())
        actual_rgb = {k for k in rgb_idx}
        actual_seg = {k for k in seg_idx}
        if actual_rgb != expected or actual_seg != expected:
            raise HeldoutError('selected member inventory changed between preflight and run')
        _atomic_json(start_marker, {'status': 'in_progress', 'started_at_unix': time.time(),
                      'preflight_binding_sha256': binding['preflight_binding_sha256'],
                      'archive_sha256': binding['archive_sha256'], 'checkpoint_sha256': binding['checkpoint_sha256'],
                      'calibration_score_vector_sha256': binding['accepted_calibration_score_vector_sha256'],
                      'protocol_sha256': binding['protocol_sha256'], 'code_sha256': binding['code_sha256'],
                      'selection_count': len(expected), 'scene_ids': list(SCENES),
                      'conditions': list(CONDITIONS), 'methods': list(METHODS),
                      'calibration_pixels_opened': True, 'held_out_pixels_opened': False,
                      'warning': 'This marker is single-use; any failure must be reported before restart.'})
        try:
            rows, cache_info = _score_and_cache(paths['rgb'], paths['class_segmentation'],
                                                rgb_idx, seg_idx, palettes, model, out_dir, binding)
            for cached_row in rows:
                cached_path = out_dir / cached_row['probability_file']
                if _sha(cached_path) != cached_row['probability_sha256']:
                    raise HeldoutError(f"cached probability digest mismatch for {cached_row['frame_id']}")
            scene_results = []
            for scene in SCENES:
                scene_results.append(_replay_cached(scene, rows, cal_scores, out_dir))
                _atomic_json(out_dir / 'partial_results.private.json',
                             {'status': 'partial', 'completed_scenes': [r['scene_id'] for r in scene_results],
                              'scene_results': scene_results,
                              'preflight_binding_sha256': binding['preflight_binding_sha256']})
            summary = summarize_two_scenes(scene_results)
            results = [arm for scene_result in scene_results for arm in scene_result['arm_results']]
            if len(results) != 42:
                raise HeldoutError(f'actual held-out replay emitted {len(results)} arms, expected 42')
            run_record = {'status': 'complete', 'scope': 'frozen descriptive VKITTI 2 held-out replay',
                          'preflight_binding_sha256': binding['preflight_binding_sha256'],
                          'archive_sha256': binding['archive_sha256'], 'checkpoint_sha256': binding['checkpoint_sha256'],
                          'calibration_original_receipt_sha256': binding['calibration_original_receipt_sha256'],
                          'calibration_scope_correction_sha256': binding['calibration_scope_correction_sha256'],
                          'calibration_score_vector_sha256': binding['accepted_calibration_score_vector_sha256'],
                          'protocol_sha256': binding['protocol_sha256'], 'code_sha256': binding['code_sha256'],
                          'runtime': binding['runtime'], 'settings_by_method': binding['settings_by_method'],
                          'selection': binding['selection'], 'selection_count': len(binding['selection']),
                          'scene_results': scene_results, 'arm_result_count': len(results),
                          'two_scene_equal_weight_summary': summary, 'cache': cache_info,
                          'censoring': {'arm_count': len(results),
                              'censored_count': sum(bool(r['metrics']['recovery_censored']) for r in results),
                              'hit_count': sum(not bool(r['metrics']['recovery_censored']) for r in results),
                              'raw_first_crossing_post_steps': [r['metrics']['first_crossing_post_step'] for r in results]},
                          'guardrail_support_by_scene_condition': [
                              {'scene': r['scene_id'], 'condition': r['condition'],
                               'pixels': r['metrics']['guardrail_pixel_support'],
                               'top1_errors': r['metrics']['guardrail_top1_error_count']}
                              for sr in scene_results for r in sr['arm_results'] if r['method'] == 'static'],
                          'limitations': ['two held-out simulated scenes only',
                              'no evidence for real-weather or Cityscapes-to-ACDC generalization',
                              'GuardRail had zero effective training pixels and zero Scene18 calibration support'],
                          }
            result_path = out_dir / 'heldout_results.private.json'
            _atomic_json(result_path, run_record)
            run_sha = _sha(result_path)
            (out_dir / 'heldout_results.private.sha256').write_text(f'{run_sha}  {result_path.name}\n')
            _atomic_json(out_dir / 'run_status.json', {'status': 'complete', 'completed_at_unix': time.time(),
                          'result_sha256': run_sha, 'arm_result_count': len(results),
                          'preflight_binding_sha256': binding['preflight_binding_sha256']})
            print(json.dumps({'status': 'complete', 'arm_result_count': len(results),
                              'censored_count': run_record['censoring']['censored_count'],
                              'hit_count': run_record['censoring']['hit_count'],
                              'results_sha256': run_sha, 'output': str(result_path)}, indent=2))
        except BaseException as exc:
            failure = {'status': 'failed_after_run_start', 'error_type': type(exc).__name__,
                       'error': str(exc), 'traceback': traceback.format_exc(),
                       'failed_at_unix': time.time(),
                       'progress': (json.loads((out_dir / 'progress.json').read_text())
                                    if (out_dir / 'progress.json').is_file() else None),
                       'preflight_binding_sha256': binding['preflight_binding_sha256'],
                       'instruction': 'Do not restart automatically; report this failed/partial run first.'}
            _atomic_json(out_dir / 'run_failure.private.json', failure)
            _atomic_json(out_dir / 'run_status.json', {'status': 'failed',
                          'failure_sha256': _sha(out_dir / 'run_failure.private.json'),
                          'preflight_binding_sha256': binding['preflight_binding_sha256']})
            raise
    finally:
        rgb_tar.close()
        seg_tar.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('preflight', 'run'))
    parser.add_argument('--out-dir', default='.local-data/vkitti2/heldout')
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    try:
        if args.phase == 'preflight':
            result = preflight(out_dir)
            print(json.dumps({'status': result['status'],
                              'preflight_binding_sha256': result['preflight_binding_sha256'],
                              'selected_frame_count': result['selected_frame_count'],
                              'synthetic_dry_run': result['synthetic_dry_run'],
                              'receipt': str(out_dir / 'preflight_receipt.json')}, indent=2))
        else:
            run_once(out_dir)
    except BaseException as exc:
        if args.phase == 'preflight':
            out_dir.mkdir(parents=True, exist_ok=True)
            _atomic_json(out_dir / 'preflight_failure.json', {'status': 'preflight_failed_no_pixels_opened',
                          'error_type': type(exc).__name__, 'error': str(exc),
                          'traceback': traceback.format_exc(), 'held_out_pixels_opened': False})
        raise


if __name__ == '__main__':
    main()

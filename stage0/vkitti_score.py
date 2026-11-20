#!/usr/bin/env python3
"""Locked native-resolution scoring, checkpoint validation, parity, and Scene18 calibration."""
import argparse
import hashlib
import io
import json
import math
import platform
import re
import sys
import time
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.confseg_core.vkitti import calibration_anchor, frame_score, restore_native_probabilities
from stage0.vkitti_train import (CLASSES, IGNORE, _model, _read_tar_bytes, preprocess, sha256,
                                 setup_determinism)

RGB_RE = re.compile(r'^Scene18/clone/frames/rgb/Camera_0/rgb_(\d+)\.jpg$')
MASK_RE = re.compile(r'^Scene18/clone/frames/classSegmentation/Camera_0/classgt_(\d+)\.png$')
PALETTE_RE = re.compile(r'^Scene18/clone/colors\.txt$')
EXPECTED_N = 339
EXPECTED_CHECKPOINT = '760e7d1a5e91fea543cd1c6aaa1946d044e23d1e96960c133360d64daf9cab4d'


class ScoreError(RuntimeError):
    pass


def _palette(textgt_path):
    palette = None
    try:
        with tarfile.open(textgt_path, 'r:gz') as archive:
            for member in archive:
                if member.isfile() and PALETTE_RE.fullmatch(member.name):
                    if palette is not None:
                        raise ScoreError('duplicate calibration palette')
                    rows = [line.split() for line in _read_tar_bytes(archive, member).decode('utf-8-sig').splitlines()
                            if line.strip()]
                    if not rows or rows[0] != ['Category', 'r', 'g', 'b']:
                        raise ScoreError('calibration palette header is invalid')
                    palette = {}
                    for row in rows[1:]:
                        if len(row) != 4:
                            raise ScoreError('calibration palette row malformed')
                        name = row[0]
                        color = tuple(map(int, row[1:]))
                        if name in palette or any(v < 0 or v > 255 for v in color):
                            raise ScoreError('calibration palette contains duplicate names or invalid channels')
                        palette[name] = color
    except (OSError, tarfile.TarError, UnicodeError, ValueError) as exc:
        if isinstance(exc, ScoreError):
            raise
        raise ScoreError(f'cannot read calibration palette ({type(exc).__name__})') from exc
    if palette is None:
        raise ScoreError('Scene18 clone palette missing')
    if len(set(palette.values())) != len(palette):
        raise ScoreError('calibration palette has ambiguous duplicate RGB values')
    if tuple(palette) != CLASSES + ('Undefined',):
        raise ScoreError('calibration palette category names/order differ from frozen model')
    return palette


def _index(path, pattern, kind):
    found = {}
    try:
        with tarfile.open(path, 'r:') as archive:
            for member in archive:
                if not member.isfile():
                    continue
                match = pattern.fullmatch(member.name)
                if not match:
                    continue
                frame = int(match.group(1))
                if frame in found:
                    raise ScoreError(f'duplicate Scene18 {kind} frame key')
                found[frame] = member
    except (OSError, tarfile.TarError) as exc:
        raise ScoreError(f'cannot index Scene18 {kind} archive ({type(exc).__name__})') from exc
    return found


def _decode(rgb_tar, mask_tar, rgb_member, mask_member, color_to_class):
    try:
        with Image.open(io.BytesIO(_read_tar_bytes(rgb_tar, rgb_member))) as src:
            rgb_image = src.convert('RGB')
            rgb = np.asarray(rgb_image, dtype=np.uint8).copy()
            native_size = rgb_image.size
        with Image.open(io.BytesIO(_read_tar_bytes(mask_tar, mask_member))) as src:
            native_mask = np.asarray(src.convert('RGB'), dtype=np.uint8).copy()
    except (OSError, ValueError) as exc:
        raise ScoreError('malformed Scene18 RGB/mask pair') from exc
    if native_size != (native_mask.shape[1], native_mask.shape[0]):
        raise ScoreError('Scene18 native RGB/mask dimensions disagree')
    labels = np.full(native_mask.shape[:2], IGNORE, dtype=np.uint8)
    known = np.zeros(native_mask.shape[:2], dtype=bool)
    for color, class_id in color_to_class.items():
        pixels = np.all(native_mask == color, axis=2)
        labels[pixels] = class_id
        known |= pixels
    if not known.all():
        raise ScoreError(f'Scene18 mask has {int((~known).sum())} unknown RGB palette pixels')
    return rgb, labels


def _load_model(weights_path, training_record_path, training_code_path, protocol_path):
    record = json.loads(Path(training_record_path).read_text())
    if record.get('status') != 'complete' or record.get('steps_completed') != 1074 or record.get('epochs_completed') != 6:
        raise ScoreError('training receipt does not record the fixed complete run')
    weight_hash = sha256(weights_path)
    if weight_hash != EXPECTED_CHECKPOINT or weight_hash != record.get('weight_file_sha256'):
        raise ScoreError('checkpoint SHA-256 differs from the accepted final model digest')
    training_code_hash = sha256(training_code_path)
    if training_code_hash != record.get('training_code_sha256'):
        raise ScoreError('training code digest no longer matches the checkpoint receipt')
    protocol = Path(protocol_path).read_text()
    if EXPECTED_CHECKPOINT not in protocol:
        raise ScoreError('calibration protocol does not bind the accepted checkpoint digest')
    model = _model().cpu()
    expected = model.state_dict()
    try:
        with np.load(weights_path, allow_pickle=False) as archive:
            state = strict_state_arrays(expected, {name: archive[name] for name in archive.files})
    except (OSError, ValueError, KeyError) as exc:
        if isinstance(exc, ScoreError):
            raise
        raise ScoreError(f'cannot safely load tensor-only checkpoint ({type(exc).__name__})') from exc
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ScoreError('strict checkpoint loading failed') from exc
    model.eval()
    return model, record, {'weights_sha256': weight_hash, 'training_code_sha256': training_code_hash,
                           'protocol_sha256': sha256(protocol_path), 'tensor_count': len(state),
                           'tensor_dtypes': sorted({str(x.dtype).replace('torch.', '') for x in state.values()})}


def strict_state_arrays(expected_state, arrays):
    """Validate a tensor-only named state before strict torch.load_state_dict."""
    if set(arrays) != set(expected_state):
        raise ScoreError('checkpoint tensor names differ from model state_dict')
    state = {}
    for name, target in expected_state.items():
        array = np.asarray(arrays[name])
        if array.dtype != target.detach().cpu().numpy().dtype or tuple(array.shape) != tuple(target.shape):
            raise ScoreError(f'checkpoint tensor shape/dtype mismatch for {name}')
        if not np.isfinite(array).all():
            raise ScoreError(f'checkpoint tensor {name} contains non-finite values')
        state[name] = torch.from_numpy(array.copy())
    return state


def _infer_one(model, rgb, labels):
    input_tensor, _, geometry = preprocess(rgb, labels)
    with torch.inference_mode():
        logits = model(input_tensor.unsqueeze(0).float().cpu())['out']
    probabilities = restore_native_probabilities(logits, geometry)
    return probabilities, geometry


def development_parity(rgb_path, seg_path, textgt_path, weights_path, training_record_path,
                       training_code_path, protocol_path, census_path, out_path):
    """Compare production native restoration/scoring to a separately expressed reference on a dev frame."""
    started = time.perf_counter()
    setup_determinism()
    model, train_record, checkpoint = _load_model(weights_path, training_record_path,
                                                    training_code_path, protocol_path)
    expected = json.loads(Path(census_path).read_text())['archive_digests']
    for key, path_key, path in (('rgb', 'rgb', rgb_path), ('class_segmentation', 'class_segmentation', seg_path),
                                ('text_metadata', None, textgt_path)):
        expected_hash = expected[path_key]['sha256'] if path_key else train_record['archive_sha256']['text_metadata']
        if sha256(path) != expected_hash:
            raise ScoreError(f'{key} archive digest differs from registered local census')
    palette = json.loads(Path(training_record_path).read_text())['publisher_palette_rgb']
    colors = {tuple(v): i for i, name in enumerate(CLASSES)
              for v in [palette[name]]}
    colors[tuple(palette['Undefined'])] = IGNORE
    # A small development-only member lookup. Only this RGB and mask are decoded.
    from stage0.vkitti_train import RGB_RE as DEV_RGB_RE, MASK_RE as DEV_MASK_RE
    def find_one(path, regex, suffix):
        with tarfile.open(path, 'r:') as tar:
            for member in tar:
                if member.isfile() and regex.fullmatch(member.name) and member.name.endswith(suffix):
                    return member
        raise ScoreError('development parity sample missing')
    with tarfile.open(rgb_path, 'r:') as rt, tarfile.open(seg_path, 'r:') as st:
        rgb_member = find_one(rgb_path, DEV_RGB_RE, 'rgb_00000.jpg')
        mask_member = find_one(seg_path, DEV_MASK_RE, 'classgt_00000.png')
        rgb, labels = _decode(rt, st, rgb_member, mask_member, colors)
    probabilities, geometry = _infer_one(model, rgb, labels)
    # Independent direct path from preprocessed input through explicit crop-before-resize.
    x, _, g2 = preprocess(rgb, labels)
    with torch.inference_mode():
        logits = model(x.unsqueeze(0).float().cpu())['out']
        crop = logits[..., g2['pad_top']:g2['pad_top'] + g2['scaled_height'],
                       g2['pad_left']:g2['pad_left'] + g2['scaled_width']]
        native_logits = F.interpolate(crop, size=(g2['native_height'], g2['native_width']),
                                       mode='bilinear', align_corners=False)
        reference = torch.softmax(native_logits, dim=1)[0].cpu().numpy()
    if geometry != g2:
        raise ScoreError('shared training/scoring preprocessing geometry differs')
    max_difference = float(np.max(np.abs(probabilities - reference)))
    if max_difference > 1e-7 or not np.allclose(probabilities.sum(axis=0), 1., rtol=0., atol=2e-6):
        raise ScoreError('development native-resolution parity failed')
    production = frame_score(probabilities, labels)
    # Hand-written score/features reference, not calling frame_score for expected values.
    p = probabilities.astype(np.float64)
    valid = labels != IGNORE
    argmax = np.argmax(p, axis=0)
    true = np.clip(labels, 0, len(CLASSES) - 1)
    candidate = np.where(argmax == labels, 0., 1. - np.take_along_axis(p, true[None], axis=0)[0])[valid]
    rank = int(valid.sum()) - int(valid.sum()) // 20
    expected_score = float(np.sort(candidate)[rank - 1])
    if production['score_tau05'] != expected_score:
        raise ScoreError('development frame score parity failed')
    entropy_ref = -np.sum(np.where(p > 0., p * np.log(np.maximum(p, 1e-300)), 0.), axis=0).mean()
    maxprob_ref = p.max(axis=0)
    feature_ref = np.array([entropy_ref, (maxprob_ref < .5).mean(), maxprob_ref.mean()])
    feature_got = np.array([production['mean_entropy'], production['low_conf_fraction'], production['mean_max_prob']])
    if not np.allclose(feature_got, feature_ref, rtol=0., atol=1e-12):
        raise ScoreError('development label-free feature parity failed')
    result = {'status': 'passed', 'scope': 'development-only', 'sample_scene': '01',
              'sample_frame_index': 0, 'native_shape': list(labels.shape),
              'geometry': geometry, 'max_probability_difference': max_difference,
              'score_tau05': production['score_tau05'], 'independent_reference_score': expected_score,
              'label_free_feature_max_abs_difference': float(np.max(np.abs(feature_got - feature_ref))),
              'valid_pixel_count': production['valid_pixel_count'],
              'guardrail_pixel_count': production['guardrail_pixel_count'],
              'guardrail_top1_error_count': production['guardrail_top1_error_count'],
              'checkpoint': checkpoint, 'calibration_or_held_out_pixels_opened': False,
              'elapsed_seconds': time.perf_counter() - started}
    Path(out_path).write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))


def calibrate(rgb_path, seg_path, textgt_path, weights_path, training_record_path,
              training_code_path, protocol_path, census_path, out_dir):
    total_started = time.perf_counter()
    setup_determinism()
    model, training_record, checkpoint = _load_model(weights_path, training_record_path,
                                                      training_code_path, protocol_path)
    census = json.loads(Path(census_path).read_text())
    archives = {'rgb': rgb_path, 'class_segmentation': seg_path, 'text_metadata': textgt_path}
    expected_digests = census['archive_digests']
    archive_digests = {}
    for name, path in archives.items():
        digest = sha256(path)
        if name == 'text_metadata':
            expected = census['publisher_metadata_sha256']
        else:
            expected = expected_digests[name]['sha256']
        if digest != expected:
            raise ScoreError(f'{name} archive digest differs from the accepted census')
        archive_digests[name] = {'sha256': digest, 'bytes': Path(path).stat().st_size}
    palette = _palette(textgt_path)
    train_palette = training_record['publisher_palette_rgb']
    if {name: list(color) for name, color in palette.items()} != train_palette:
        raise ScoreError('Scene18 publisher palette differs from frozen training palette')
    color_to_class = {tuple(palette[name]): i for i, name in enumerate(CLASSES)}
    color_to_class[tuple(palette['Undefined'])] = IGNORE
    rgb_idx = _index(rgb_path, RGB_RE, 'RGB')
    mask_idx = _index(seg_path, MASK_RE, 'mask')
    expected_frames = set(range(EXPECTED_N))
    if set(rgb_idx) != expected_frames or set(mask_idx) != expected_frames:
        raise ScoreError('Scene18 clone calibration must contain all exact frame keys 0–338')
    if any(not (x.startswith('Scene01/clone/') or x.startswith('Scene06/clone/'))
           for x in training_record['ordered_training_ids']):
        raise ScoreError('training receipt contains a frame outside scenes 01/06 clone')
    frame_rows, scores, features = [], [], []
    class_counts = np.zeros(len(CLASSES), dtype=np.int64)
    class_errors = np.zeros(len(CLASSES), dtype=np.int64)
    valid_counts, top1_errors = [], []
    inference_started = time.perf_counter()
    with tarfile.open(rgb_path, 'r:') as rt, tarfile.open(seg_path, 'r:') as st:
        for frame in range(EXPECTED_N):
            rgb, labels = _decode(rt, st, rgb_idx[frame], mask_idx[frame], color_to_class)
            probs, geometry = _infer_one(model, rgb, labels)
            result = frame_score(probs, labels)
            score = result.pop('score_tau05')
            pixel_scores = result.pop('pixel_scores')
            # Image-level finite checks and fixed preprocessing resolution.
            if probs.shape[0] != 14 or not np.isfinite(pixel_scores).all() or result['valid_pixel_count'] <= 0:
                raise ScoreError(f'calibration frame {frame} has invalid score/probability support')
            class_counts += result['native_class_pixel_counts']
            class_errors += result['top1_error_counts_by_class']
            valid_counts.append(result['valid_pixel_count'])
            top1_errors.append(result['top1_error_count'])
            scores.append(score)
            feat = [result['mean_entropy'], result['low_conf_fraction'], result['mean_max_prob']]
            features.append(feat)
            frame_rows.append({'frame_index': frame,
                               'frame_id': f'Scene18/clone/Camera_0/{frame:05}',
                               'score_tau05': score, 'valid_pixel_count': result['valid_pixel_count'],
                               'native_class_pixel_counts': result['native_class_pixel_counts'].tolist(),
                               'top1_error_count': result['top1_error_count'],
                               'top1_error_counts_by_class': result['top1_error_counts_by_class'].tolist(),
                               'guardrail_pixel_count': result['guardrail_pixel_count'],
                               'guardrail_top1_error_count': result['guardrail_top1_error_count'],
                               'label_free_features': feat, 'preprocessing_geometry': geometry})
    inference_seconds = time.perf_counter() - inference_started
    scores = np.asarray(scores, dtype='<f8')
    features = np.asarray(features, dtype='<f8')
    anchor = calibration_anchor(scores, .10)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    score_file = out_dir / 'scene18_clone_scores_tau05.npy'
    feature_file = out_dir / 'scene18_clone_label_free_features.npy'
    np.save(score_file, scores, allow_pickle=False)
    np.save(feature_file, features, allow_pickle=False)
    vector_digest = hashlib.sha256(scores.astype('<f8', copy=False).tobytes()).hexdigest()
    names = list(CLASSES)
    class_support = {name: int(count) for name, count in zip(names, class_counts)}
    class_top1_errors = {name: int(count) for name, count in zip(names, class_errors)}
    class_top1_error_rates = {name: (float(err / support) if support else None)
                              for name, err, support in zip(names, class_errors, class_counts)}
    record = {'status': 'complete', 'phase': 'calibration_only',
              'dataset': 'Virtual KITTI 2 v2.0.3', 'scene': '18', 'condition': 'clone',
              'scope': 'calibration', 'camera': 0, 'frame_key_range': [0, 338],
              'frame_count': len(scores), 'ordered_frame_ids': [row['frame_id'] for row in frame_rows],
              'training_scenes': ['01', '06'], 'calibration_training_scene_disjoint': True,
              'held_out_evaluation_scenes': ['02', '20'], 'calibration_evaluation_scene_disjoint': True,
              'calibration_or_held_out_pixels_opened': False,
              'protocol_sha256': checkpoint['protocol_sha256'],
              'archive_digests': archive_digests, 'checkpoint': checkpoint,
              'training_record_sha256': sha256(training_record_path),
              'score_code_sha256': sha256(__file__),
              'class_names_in_logit_order': names, 'ignore_index': IGNORE,
              'palette_rgb': {name: list(color) for name, color in palette.items()},
              'preprocessing': training_record['preprocessing'],
              'score_definition': 'correct top-1=0; otherwise 1-p_true; rank n-floor(n/20) ascending, one-based',
              'risk_alpha': .10, 'miss_fraction_tau': .05,
              'calibration_rank_rule': 'ceil((n+1)*(1-alpha)); weak acceptance at equality',
              'calibration_anchor': anchor['anchor'], 'anchor_rank_one_based': anchor['rank_one_based'],
              'calibration_sample_size': anchor['n'], 'anchor_clipped_to_one': anchor['clipped_to_one'],
              'score_vector_sha256': vector_digest, 'score_file': score_file.name,
              'score_file_sha256': sha256(score_file), 'feature_file': feature_file.name,
              'feature_file_sha256': sha256(feature_file),
              'valid_pixel_count_by_frame_min': int(min(valid_counts)),
              'valid_pixel_count_by_frame_max': int(max(valid_counts)),
              'valid_pixel_count_total': int(sum(valid_counts)),
              'top1_error_count_total': int(sum(top1_errors)),
              'native_class_pixel_support': class_support,
              'top1_errors_by_class': class_top1_errors,
              'top1_error_rate_by_class': class_top1_error_rates,
              'guardrail_pixel_support': class_support['GuardRail'],
              'guardrail_top1_error_count': class_top1_errors['GuardRail'],
              'score_greater_than_anchor_count': int(np.sum(scores > anchor['anchor'])),
              'frame_results_file': 'scene18_clone_frame_results.json',
              'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                          'torchvision': torchvision.__version__, 'numpy': np.__version__,
                          'pillow': Image.__version__, 'platform': platform.platform(),
                          'device': 'cpu', 'threads': torch.get_num_threads(),
                          'interop_threads': torch.get_num_interop_threads(),
                          'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                          'mps_available': bool(torch.backends.mps.is_available())},
              'inference_elapsed_seconds': inference_seconds,
              'total_elapsed_seconds_including_hashes_and_setup': time.perf_counter() - total_started,
              'frames_per_second_inference_only': len(scores) / inference_seconds,
              'coverage_statement': 'split conformal-style calibration rank; no population coverage guarantee for dependent frames or shifted renders'}
    (out_dir / 'scene18_clone_frame_results.json').write_text(json.dumps(frame_rows, separators=(',', ':'), allow_nan=False) + '\n')
    (out_dir / 'calibration_receipt.json').write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({k: record[k] for k in ('status', 'frame_count', 'calibration_anchor',
                       'anchor_rank_one_based', 'score_vector_sha256', 'guardrail_pixel_support',
                       'guardrail_top1_error_count', 'inference_elapsed_seconds',
                       'frames_per_second_inference_only')}, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('dev-parity', 'calibrate'))
    parser.add_argument('--rgb', required=True)
    parser.add_argument('--segmentation', required=True)
    parser.add_argument('--textgt', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--training-record', required=True)
    parser.add_argument('--training-code', default='stage0/vkitti_train.py')
    parser.add_argument('--protocol', default='project/vkitti_calibration_replay_protocol.md')
    parser.add_argument('--census', default='.local-data/vkitti2/census.json')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.phase == 'dev-parity':
        development_parity(args.rgb, args.segmentation, args.textgt, args.weights,
                           args.training_record, args.training_code, args.protocol,
                           args.census, args.out)
    else:
        calibrate(args.rgb, args.segmentation, args.textgt, args.weights,
                  args.training_record, args.training_code, args.protocol,
                  args.census, args.out)


if __name__ == '__main__':
    try:
        main()
    except ScoreError as exc:
        raise SystemExit(f'VKITTI score pipeline halted: {exc}')

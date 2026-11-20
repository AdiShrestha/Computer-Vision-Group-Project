#!/usr/bin/env python3
"""Fixed, development-only Virtual KITTI 2 segmentation preflight and training."""
import argparse
import hashlib
import io
import json
import math
import platform
import random
import re
import time
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

SEED = 101
SCENES = ('01', '06')
CONDITION = 'clone'
NATIVE_COUNT = {'01': 447, '06': 270}
CLASSES = ('Terrain', 'Sky', 'Tree', 'Vegetation', 'Building', 'Road', 'GuardRail',
           'TrafficSign', 'TrafficLight', 'Pole', 'Misc', 'Truck', 'Car', 'Van')
IGNORE = 255
HEIGHT, WIDTH = 160, 512
RGB_RE = re.compile(r'^Scene(01|06)/clone/frames/rgb/Camera_0/rgb_(\d+)\.jpg$')
MASK_RE = re.compile(r'^Scene(01|06)/clone/frames/classSegmentation/Camera_0/classgt_(\d+)\.png$')
COLOR_RE = re.compile(r'^Scene(01|06)/clone/colors\.txt$')


class TrainingError(RuntimeError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def setup_determinism():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def _read_tar_bytes(tar, member):
    stream = tar.extractfile(member)
    if stream is None:
        raise TrainingError('selected development TAR member could not be read')
    return stream.read()


def inventory(rgb_path, seg_path, textgt_path):
    """Index headers for clone development files, then enforce one-to-one native keys."""
    rgb, seg, colors = {}, {}, {}
    try:
        with tarfile.open(rgb_path, 'r:') as archive:
            for member in archive:
                if not member.isfile():
                    continue
                match = RGB_RE.fullmatch(member.name)
                if match:
                    scene, frame = match.groups()
                    key = (scene, int(frame))
                    if key in rgb:
                        raise TrainingError('duplicate development RGB frame key')
                    rgb[key] = member
        with tarfile.open(seg_path, 'r:') as archive:
            for member in archive:
                if not member.isfile():
                    continue
                match = MASK_RE.fullmatch(member.name)
                if match:
                    scene, frame = match.groups()
                    key = (scene, int(frame))
                    if key in seg:
                        raise TrainingError('duplicate development mask frame key')
                    seg[key] = member
        with tarfile.open(textgt_path, 'r:gz') as archive:
            for member in archive:
                if not member.isfile():
                    continue
                match = COLOR_RE.fullmatch(member.name)
                if match:
                    scene = match.group(1)
                    if scene in colors:
                        raise TrainingError('duplicate development palette')
                    data = _read_tar_bytes(archive, member).decode('utf-8-sig')
                    rows = [line.split() for line in data.splitlines() if line.strip()]
                    if not rows or rows[0] != ['Category', 'r', 'g', 'b']:
                        raise TrainingError('development palette header is invalid')
                    palette = {}
                    for row in rows[1:]:
                        if len(row) != 4:
                            raise TrainingError('development palette row is malformed')
                        name = row[0]
                        try:
                            color = tuple(map(int, row[1:]))
                        except ValueError as exc:
                            raise TrainingError('development palette channel is not an integer') from exc
                        if name in palette or any(c < 0 or c > 255 for c in color):
                            raise TrainingError('development palette has duplicate names or invalid RGB')
                        palette[name] = color
                    if len(set(palette.values())) != len(palette):
                        raise TrainingError('development palette contains ambiguous duplicate RGB values')
                    colors[scene] = palette
    except (OSError, tarfile.TarError, UnicodeError) as exc:
        raise TrainingError(f'cannot index authorized development archive ({type(exc).__name__})') from exc
    if set(colors) != set(SCENES):
        raise TrainingError('both development scene palettes are required')
    reference = colors['01']
    if colors['06'] != reference:
        raise TrainingError('development scene palettes differ')
    if tuple(reference) != CLASSES + ('Undefined',):
        raise TrainingError('publisher palette category names/order do not match the registered mapping')
    colors_by_id = [reference[name] for name in CLASSES]
    undefined = reference.get('Undefined')
    if undefined is None or undefined in colors_by_id:
        raise TrainingError('Undefined color is missing or ambiguous')
    if set(rgb) != set(seg):
        raise TrainingError(f'development RGB/mask keys do not match: RGB={len(rgb)}, masks={len(seg)}')
    for scene, expected in NATIVE_COUNT.items():
        selected = [key for key in rgb if key[0] == scene]
        if len(selected) != expected:
            raise TrainingError(f'development scene {scene} has {len(selected)} pairs; expected {expected}')
        if {frame for s, frame in selected} != set(range(expected)):
            raise TrainingError(f'development scene {scene} frame keys are incomplete or noncontiguous')
    if len(rgb) != 717:
        raise TrainingError(f'expected exactly 717 development pairs, got {len(rgb)}')
    return rgb, seg, reference, colors_by_id, undefined


def _decode_pair(rgb_tar, seg_tar, rgb_member, mask_member, color_to_class, undefined):
    try:
        with Image.open(io.BytesIO(_read_tar_bytes(rgb_tar, rgb_member))) as source:
            rgb_native = source.convert('RGB')
            rgb_size = rgb_native.size
            rgb = np.asarray(rgb_native, dtype=np.uint8)
        with Image.open(io.BytesIO(_read_tar_bytes(seg_tar, mask_member))) as source:
            mask_rgb = np.asarray(source.convert('RGB'), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise TrainingError('malformed selected development image or mask') from exc
    if rgb_size != (mask_rgb.shape[1], mask_rgb.shape[0]):
        raise TrainingError('native RGB/mask dimensions do not agree')
    labels = np.full(mask_rgb.shape[:2], IGNORE, dtype=np.uint8)
    known = np.zeros(mask_rgb.shape[:2], dtype=bool)
    for color, class_id in color_to_class.items():
        pixels = np.all(mask_rgb == color, axis=2)
        labels[pixels] = IGNORE if color == undefined else class_id
        known |= pixels
    if not known.all():
        bad = int((~known).sum())
        raise TrainingError(f'development mask has {bad} unknown/ambiguous RGB pixels')
    return rgb, labels


def preprocess(rgb, labels):
    h, w = labels.shape
    scale = min(WIDTH / w, HEIGHT / h)
    nw = max(1, min(WIDTH, int(math.floor(w * scale + 0.5))))
    nh = max(1, min(HEIGHT, int(math.floor(h * scale + 0.5))))
    left, top = (WIDTH - nw) // 2, (HEIGHT - nh) // 2
    right, bottom = WIDTH - nw - left, HEIGHT - nh - top
    rgb_image = Image.fromarray(rgb, 'RGB').resize((nw, nh), Image.Resampling.BILINEAR)
    rgb_canvas = Image.new('RGB', (WIDTH, HEIGHT), (0, 0, 0))
    rgb_canvas.paste(rgb_image, (left, top))
    rgb_tensor = np.asarray(rgb_canvas, dtype=np.float32).copy() / 255.0
    label_image = Image.fromarray(labels, 'L').resize((nw, nh), Image.Resampling.NEAREST)
    label_canvas = Image.new('L', (WIDTH, HEIGHT), IGNORE)
    label_canvas.paste(label_image, (left, top))
    label_tensor = np.asarray(label_canvas, dtype=np.uint8).copy()
    geometry = {'native_width': w, 'native_height': h, 'scaled_width': nw,
                'scaled_height': nh, 'pad_left': left, 'pad_top': top,
                'pad_right': right, 'pad_bottom': bottom}
    return torch.from_numpy(rgb_tensor).permute(2, 0, 1), torch.from_numpy(label_tensor), geometry


def _model():
    return deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None,
                                        num_classes=len(CLASSES), aux_loss=False)


def _private_run_context(rgb_path, seg_path, textgt_path, ids, code_path, palette, undefined):
    return {'dataset': 'Virtual KITTI 2 v2.0.3',
            'archive_sha256': {'rgb': sha256(rgb_path), 'class_segmentation': sha256(seg_path),
                               'text_metadata': sha256(textgt_path)},
            'ordered_training_ids': ids, 'training_code_sha256': sha256(code_path),
            'seed': SEED, 'class_names_in_logit_order': list(CLASSES),
            'publisher_palette_rgb': {name: [int(x) for x in color] for name, color in palette.items()},
            'undefined_rgb': list(undefined),
            'preprocessing': {'canvas_width': WIDTH, 'canvas_height': HEIGHT,
                              'scale': 'min(512/width,160/height)', 'dimension_rounding': 'nearest, half up; min 1',
                              'centering': 'floor half to left/top; odd remainder right/bottom',
                              'rgb_resize': 'Pillow bilinear', 'rgb_fill': [0, 0, 0],
                              'rgb_normalization': 'float32 RGB / 255', 'mask_resize': 'Pillow nearest',
                              'mask_fill': IGNORE, 'undefined_to_ignore': IGNORE},
            'resolved_settings': {'architecture': 'torchvision DeepLabV3 MobileNetV3-Large',
                                  'weights': None, 'weights_backbone': None,
                                  'aux_loss': False, 'num_classes': len(CLASSES),
                                  'epochs': 6, 'batch_size': 4, 'remainder': 'merge singleton into preceding batch',
                                  'steps_per_epoch': 179, 'total_steps': 1074,
                                  'optimizer': 'AdamW', 'learning_rate': .001,
                                  'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': .01,
                                  'loss': 'unweighted cross-entropy mean over valid pixels',
                                  'scheduler': None, 'augmentations': None,
                                  'device': 'cpu', 'dtype': 'float32', 'torch_threads': 4,
                                  'torch_interop_threads': 1, 'deterministic_algorithms': True},
            'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                        'torchvision': torchvision.__version__, 'numpy': np.__version__,
                        'pillow': Image.__version__, 'platform': platform.platform(),
                        'torch_default_dtype': str(torch.get_default_dtype()),
                        'mps_available': bool(torch.backends.mps.is_available())}}


def inspect_and_preflight(rgb_path, seg_path, textgt_path, out_path):
    preflight_started = time.perf_counter()
    rgb_index, seg_index, palette, class_colors, undefined = inventory(rgb_path, seg_path, textgt_path)
    color_to_class = {color: idx for idx, color in enumerate(class_colors)}
    color_to_class[undefined] = IGNORE
    ids = [f'Scene{s}/clone/Camera_0/{frame:05}' for s, frame in sorted(rgb_index)]
    counts_by_scene = {scene: np.zeros(len(CLASSES), dtype=np.int64) for scene in SCENES}
    geometry_by_scene = {scene: None for scene in SCENES}
    # Validate all training masks and count the pixels the fixed resize actually presents to loss.
    with tarfile.open(rgb_path, 'r:') as rgb_tar, tarfile.open(seg_path, 'r:') as seg_tar:
        for key in sorted(rgb_index):
            image, labels = _decode_pair(rgb_tar, seg_tar, rgb_index[key], seg_index[key],
                                          color_to_class, undefined)
            _, label_t, geometry = preprocess(image, labels)
            if not (label_t != IGNORE).any():
                raise TrainingError('development training mask is empty after transformation')
            c = np.bincount(label_t.numpy()[label_t.numpy() != IGNORE], minlength=len(CLASSES))
            counts_by_scene[key[0]] += c
            previous = geometry_by_scene[key[0]]
            if previous is None:
                geometry_by_scene[key[0]] = geometry
            elif previous != geometry:
                raise TrainingError('inconsistent image geometry within development scene')
    # Fresh model/optimizer state for preflight; the full training phase restarts from seed 101.
    setup_determinism()
    model = _model().cpu().train()
    with tarfile.open(rgb_path, 'r:') as rgb_tar, tarfile.open(seg_path, 'r:') as seg_tar:
        batch_keys = sorted(rgb_index)[:4]
        pairs = [_decode_pair(rgb_tar, seg_tar, rgb_index[k], seg_index[k], color_to_class, undefined)
                 for k in batch_keys]
    tensors = [preprocess(*pair)[:2] for pair in pairs]
    x = torch.stack([pair[0] for pair in tensors])
    y = torch.stack([pair[1] for pair in tensors]).long()
    model.zero_grad(set_to_none=True)
    logits = model(x)['out']
    loss = F.cross_entropy(logits, y, ignore_index=IGNORE, reduction='mean')
    loss.backward()
    gradients_finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    if logits.shape != (4, len(CLASSES), HEIGHT, WIDTH) or not torch.isfinite(logits).all() or not torch.isfinite(loss) or not gradients_finite:
        raise TrainingError('development forward/backward preflight returned non-finite values or wrong shape')
    record = _private_run_context(rgb_path, seg_path, textgt_path, ids, Path(__file__), palette, undefined)
    record.update({'phase': 'preflight_only', 'status': 'ok',
                   'development_pair_count': len(ids), 'pairs_by_scene': NATIVE_COUNT,
                   'verified_palettes_equal': True, 'native_rgb_mask_geometry_agreement': True,
                   'post_resize_class_pixel_counts_by_scene': {
                       scene: {name: int(count) for name, count in zip(CLASSES, counts_by_scene[scene])}
                       for scene in SCENES},
                   'preflight_batch_shape': list(x.shape), 'preflight_logits_shape': list(logits.shape),
                   'preflight_loss': float(loss.detach()), 'finite_forward_backward': True,
                   'preflight_elapsed_seconds_including_archive_hashes': time.perf_counter() - preflight_started,
                   'calibration_or_held_out_pixels_opened': False})
    Path(out_path).write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'status': 'ok', 'phase': 'preflight_only', 'pairs': len(ids),
                      'preflight_loss': float(loss.detach()), 'logits_shape': list(logits.shape),
                      'report': str(out_path)}, indent=2))
    return (rgb_index, seg_index, palette, color_to_class, undefined, record)


def train(rgb_path, seg_path, textgt_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_index, seg_index, palette, class_colors, undefined = inventory(rgb_path, seg_path, textgt_path)
    color_to_class = {color: idx for idx, color in enumerate(class_colors)}
    color_to_class[undefined] = IGNORE
    ids = [f'Scene{s}/clone/Camera_0/{frame:05}' for s, frame in sorted(rgb_index)]
    record = _private_run_context(rgb_path, seg_path, textgt_path, ids, Path(__file__), palette, undefined)
    setup_determinism()  # fresh restart after the separate preflight process
    model = _model().cpu().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=.01)
    permutation_generator = torch.Generator(device='cpu').manual_seed(SEED)
    epoch_losses, epoch_seconds = [], []
    start_all = time.perf_counter()
    steps = 0
    with tarfile.open(rgb_path, 'r:') as rgb_tar, tarfile.open(seg_path, 'r:') as seg_tar:
        for epoch in range(6):
            epoch_start = time.perf_counter()
            permutation = torch.randperm(len(ids), generator=permutation_generator).tolist()
            batches = [permutation[i:i + 4] for i in range(0, len(permutation), 4)]
            if len(batches[-1]) == 1:
                batches[-2].extend(batches.pop())
            if len(batches) != 179 or max(map(len, batches)) != 5 or min(map(len, batches)) != 4:
                raise TrainingError('registered batch construction did not produce 179 non-singleton steps')
            loss_total, valid_pixels = 0., 0
            for batch_indices in batches:
                batch_keys = [sorted(rgb_index)[i] for i in batch_indices]
                pairs = [_decode_pair(rgb_tar, seg_tar, rgb_index[k], seg_index[k], color_to_class, undefined)
                         for k in batch_keys]
                tensors = [preprocess(*pair)[:2] for pair in pairs]
                x = torch.stack([pair[0] for pair in tensors])
                y = torch.stack([pair[1] for pair in tensors]).long()
                nvalid = int((y != IGNORE).sum())
                if nvalid == 0:
                    raise TrainingError('empty valid training batch after transformation')
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)['out']
                loss = F.cross_entropy(logits, y, ignore_index=IGNORE, reduction='mean')
                if not torch.isfinite(loss):
                    raise TrainingError(f'non-finite loss at epoch {epoch + 1}, step {steps + 1}')
                loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise TrainingError(f'non-finite gradient at epoch {epoch + 1}, step {steps + 1}')
                optimizer.step()
                if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                    raise TrainingError(f'non-finite parameter at epoch {epoch + 1}, step {steps + 1}')
                loss_total += float(loss.detach()) * nvalid
                valid_pixels += nvalid
                steps += 1
            elapsed = time.perf_counter() - epoch_start
            epoch_losses.append(loss_total / valid_pixels)
            epoch_seconds.append(elapsed)
            print(json.dumps({'epoch': epoch + 1, 'steps': steps, 'mean_valid_pixel_loss': epoch_losses[-1],
                              'elapsed_seconds': elapsed}, sort_keys=True), flush=True)
    elapsed_all = time.perf_counter() - start_all
    if steps != 1074 or len(epoch_losses) != 6:
        raise TrainingError(f'training stopped before the registered 1,074 steps ({steps} completed)')
    state = model.state_dict()
    npz_path = out_dir / 'vkitti2_deeplab_mobilenetv3_seed101_epoch6.npz'
    np.savez(npz_path, **{k: v.detach().cpu().numpy() for k, v in state.items()})
    if any(not np.isfinite(arr).all() for arr in (v.detach().cpu().numpy() for v in state.values())):
        raise TrainingError('final weights contain non-finite tensors')
    counts = {scene: {name: 0 for name in CLASSES} for scene in SCENES}
    # The preflight census is repeated to bind effective transformed training-pixel support.
    with tarfile.open(rgb_path, 'r:') as rgb_tar, tarfile.open(seg_path, 'r:') as seg_tar:
        for key in sorted(rgb_index):
            image, labels = _decode_pair(rgb_tar, seg_tar, rgb_index[key], seg_index[key], color_to_class, undefined)
            _, label_t, _ = preprocess(image, labels)
            labels_np = label_t.numpy()
            values = np.bincount(labels_np[labels_np != IGNORE], minlength=len(CLASSES))
            for name, value in zip(CLASSES, values):
                counts[key[0]][name] += int(value)
    record.update({'phase': 'fixed_training', 'status': 'complete',
                   'epochs_completed': 6, 'steps_completed': steps, 'steps_per_epoch': [179] * 6,
                   'epoch_mean_valid_pixel_loss': epoch_losses, 'epoch_elapsed_seconds': epoch_seconds,
                   'elapsed_seconds_total_training_only': elapsed_all,
                   'images_per_second': 717 * 6 / elapsed_all,
                   'optimizer_steps_per_second': steps / elapsed_all,
                   'post_resize_class_pixel_counts_by_scene': counts,
                   'weight_file': npz_path.name, 'weight_file_sha256': sha256(npz_path),
                   'weight_file_bytes': npz_path.stat().st_size,
                   'weight_tensor_count': len(state),
                   'weight_tensor_dtypes': sorted({str(v.dtype).replace('torch.', '') for v in state.values()}),
                   'weight_tensor_shapes': {k: list(v.shape) for k, v in state.items()},
                   'calibration_or_held_out_pixels_opened': False,
                   'training_only_metrics': {'mean_iou': 'not computed; no fit-set predictions made',
                                             'majority_baseline': 'not computed'}})
    (out_dir / 'training_record.json').write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'status': 'complete', 'steps': steps, 'elapsed_seconds': elapsed_all,
                      'images_per_second': record['images_per_second'],
                      'weight_file': str(npz_path), 'weight_sha256': record['weight_file_sha256']}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('preflight', 'train'))
    parser.add_argument('--rgb', required=True)
    parser.add_argument('--segmentation', required=True)
    parser.add_argument('--textgt', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.phase == 'preflight':
        inspect_and_preflight(args.rgb, args.segmentation, args.textgt, args.out)
    else:
        train(args.rgb, args.segmentation, args.textgt, args.out)


if __name__ == '__main__':
    try:
        main()
    except TrainingError as exc:
        raise SystemExit(f'VKITTI training halted: {exc}')

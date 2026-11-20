"""Native-resolution Virtual KITTI 2 scoring and split-calibration primitives."""
import math

import numpy as np


def frame_score(probabilities, labels, *, alpha_miss=0.05, n_classes=14, ignore_index=255):
    """Compute S(tau) and label-free features from K×H×W probabilities and native labels."""
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels)
    if p.ndim != 3 or p.shape[0] != n_classes or y.shape != p.shape[1:]:
        raise ValueError('probability/label shape mismatch; expected KxHxW and HxW')
    if y.dtype.kind not in 'iu':
        raise ValueError('labels must use an integer dtype')
    if not 0 <= alpha_miss < 1:
        raise ValueError('alpha_miss must be in [0,1)')
    if not np.isfinite(p).all() or np.any((p < 0.) | (p > 1.)):
        raise ValueError('probabilities must be finite and lie in [0,1]')
    if not np.allclose(p.sum(axis=0), 1., rtol=0., atol=2e-5):
        raise ValueError('probabilities must sum to one per pixel')
    bad_labels = (y != ignore_index) & ((y < 0) | (y >= n_classes))
    if bad_labels.any():
        raise ValueError('label outside registered class mapping')
    valid = y != ignore_index
    n_valid = int(valid.sum())
    if not n_valid:
        raise ValueError('all-void frame blocks the fixed calibration/replay window')
    # np.argmax returns the first/lowest class index on ties.
    top1 = np.argmax(p, axis=0)
    p_true = np.take_along_axis(p, np.clip(y, 0, n_classes - 1)[None, ...], axis=0)[0]
    pixel_scores = np.where(top1 == y, 0., 1. - p_true)[valid]
    rank = n_valid - math.floor(n_valid * alpha_miss)
    score = float(np.partition(pixel_scores, rank - 1)[rank - 1])
    counts = np.bincount(y[valid].astype(np.int64), minlength=n_classes)
    errors = valid & (top1 != y)
    top1_errors = np.bincount(y[errors].astype(np.int64), minlength=n_classes)
    # Compute entropy in natural units on all pixels; 0*log(0) is defined as zero.
    plogp = np.zeros_like(p)
    positive = p > 0.
    plogp[positive] = p[positive] * np.log(p[positive])
    entropy = -plogp.sum(axis=0)
    maxprob = p.max(axis=0)
    return {'score_tau05': score, 'valid_pixel_count': n_valid,
            'native_class_pixel_counts': counts.astype(np.int64),
            'top1_error_count': int(errors.sum()),
            'top1_error_counts_by_class': top1_errors.astype(np.int64),
            'guardrail_pixel_count': int(counts[6]),
            'guardrail_top1_error_count': int(top1_errors[6]),
            'mean_entropy': float(entropy.mean()),
            'low_conf_fraction': float((maxprob < 0.5).mean()),
            'mean_max_prob': float(maxprob.mean()),
            'pixel_scores': pixel_scores}


def calibration_rank(n, alpha=0.10):
    if int(n) != n or n < 1 or not 0 < alpha < 1:
        raise ValueError('invalid calibration sample size or risk level')
    return int(math.ceil((int(n) + 1) * (1. - alpha) - 1e-12))


def calibration_anchor(scores, alpha=0.10):
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all() or np.any((x < 0.) | (x > 1.)):
        raise ValueError('calibration scores must be a finite nonempty vector in [0,1]')
    rank = calibration_rank(len(x), alpha)
    if rank > len(x):
        return {'anchor': 1., 'rank_one_based': rank, 'n': len(x), 'clipped_to_one': True}
    return {'anchor': float(np.partition(x, rank - 1)[rank - 1]),
            'rank_one_based': rank, 'n': len(x), 'clipped_to_one': False}


def set_size_statistics(probabilities, threshold):
    """Exact native-pixel set size and multi-class fraction with top-1 fallback."""
    # Keep the native float32 probability values exactly representable, then do
    # activation subtraction/comparison in float64. Subtracting in float32 can
    # round an activation down across a float64 controller threshold boundary.
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 3 or not np.isfinite(p).all() or np.any((p < 0.) | (p > 1.)):
        raise ValueError('native probabilities must be finite KxHxW values in [0,1]')
    if not np.isfinite(threshold) or not 0. <= threshold <= 1.:
        raise ValueError('set threshold must be finite in [0,1]')
    members = (1. - p) <= float(threshold)
    empty = ~members.any(axis=0)
    top1 = np.argmax(p, axis=0)
    if empty.any():
        rows, cols = np.nonzero(empty)
        members[top1[rows, cols], rows, cols] = True
    sizes = members.sum(axis=0)
    return {'mean_set_size': float(sizes.mean()),
            'fraction_multi_class': float((sizes > 1).mean()),
            'pixel_count': int(sizes.size)}


def restore_native_probabilities(logits, geometry):
    """Crop letterbox padding then resize logits to native pixels before softmax."""
    import torch
    import torch.nn.functional as F
    x = torch.as_tensor(logits, dtype=torch.float32)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4 or x.shape[0] != 1 or tuple(x.shape[-2:]) != (160, 512):
        raise ValueError('logits must be 1xKx160x512 (or Kx160x512)')
    if not torch.isfinite(x).all():
        raise ValueError('model logits must be finite')
    top, left = int(geometry['pad_top']), int(geometry['pad_left'])
    height, width = int(geometry['scaled_height']), int(geometry['scaled_width'])
    nh, nw = int(geometry['native_height']), int(geometry['native_width'])
    if min(top, left) < 0 or min(height, width, nh, nw) < 1 or top + height > 160 or left + width > 512:
        raise ValueError('invalid preprocessing geometry')
    cropped = x[..., top:top + height, left:left + width]
    native_logits = F.interpolate(cropped, size=(nh, nw), mode='bilinear', align_corners=False)
    probabilities = torch.softmax(native_logits, dim=1)[0]
    if not torch.isfinite(probabilities).all():
        raise ValueError('native probabilities are non-finite')
    return probabilities.detach().cpu().numpy()

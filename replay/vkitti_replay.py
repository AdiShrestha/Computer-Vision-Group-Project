"""Locked 30-frame clear-prefix / 30-frame render-suffix causal replay primitives."""
import hashlib
import platform
import json
import math
from pathlib import Path

import numpy as np

from core.confseg_core import CausalController, FeedbackReplay
from core.confseg_core.vkitti import calibration_anchor, set_size_statistics

METHODS = ('static', 'aci', 'qt', 'rolling', 'pi', 'history_pid', 'feature_pid')
CONDITIONS = ('clone', 'fog', 'rain')
PREFIX = 30
SUFFIX = 30
ALPHA = 0.10
RISK_WINDOW = 10
RECOVERY_RISK = 0.15
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
RUNTIME_IDENTITY = {'python': platform.python_version(), 'implementation': platform.python_implementation(),
                    'platform': platform.platform(), 'numpy': np.__version__}


class ReplayProtocolError(ValueError):
    pass


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _validate_vector(values, name):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) != PREFIX or not np.isfinite(x).all() or np.any((x < 0.) | (x > 1.)):
        raise ReplayProtocolError(f'{name} must be exactly 30 finite values in [0,1]')
    return x


def _validate_features(values, name):
    x = np.asarray(values, dtype=np.float64)
    if x.shape != (PREFIX, 3) or not np.isfinite(x).all():
        raise ReplayProtocolError(f'{name} must be a finite 30x3 current-frame feature matrix')
    return x


def _metrics(scores, thresholds, boundary_jump):
    errors = (np.asarray(scores) > np.asarray(thresholds)).astype(np.float64)
    if len(errors) != SUFFIX:
        raise ReplayProtocolError('post-switch score/threshold lengths must be 30')
    rolling = np.array([errors[max(0, i - RISK_WINDOW + 1):i + 1].mean()
                        for i in range(SUFFIX)], dtype=np.float64)
    crossing = next((i for i in range(RISK_WINDOW - 1, SUFFIX) if rolling[i] <= RECOVERY_RISK), None)
    hit_step = None if crossing is None else crossing + 1
    restricted = min(hit_step if hit_step is not None else SUFFIX, SUFFIX)
    within_tv = float(np.abs(np.diff(thresholds)).mean()) if len(thresholds) > 1 else 0.
    return {'post_switch_breach_rate': float(errors.mean()),
            'post_switch_risk_excess': float(max(0., errors.mean() - ALPHA)),
            'post_switch_rolling_risk': rolling.tolist(),
            'first_crossing_post_step': hit_step, 'recovery_censored': hit_step is None,
            'restricted_log_endpoint': float(math.log1p(restricted)),
            'normalized_area_above_risk_010': float(np.maximum(0., rolling - ALPHA).mean()),
            'within_suffix_mean_absolute_threshold_change': within_tv,
            'switch_boundary_threshold_jump': float(boundary_jump)}


def _set_metrics(thresholds, probability_provider):
    if probability_provider is None:
        return {'native_mean_set_size': None, 'native_fraction_multi_class': None,
                'set_size_pixel_count': 0, 'set_size_status': 'not supplied'}
    weighted_size = weighted_multi = total_pixels = 0
    for j, threshold in enumerate(thresholds):
        stats = set_size_statistics(probability_provider(PREFIX + j), float(threshold))
        weighted_size += stats['mean_set_size'] * stats['pixel_count']
        weighted_multi += stats['fraction_multi_class'] * stats['pixel_count']
        total_pixels += stats['pixel_count']
    if not total_pixels:
        raise ReplayProtocolError('native set-size provider returned no pixels')
    return {'native_mean_set_size': float(weighted_size / total_pixels),
            'native_fraction_multi_class': float(weighted_multi / total_pixels),
            'set_size_pixel_count': int(total_pixels), 'set_size_status': 'exact native pixels'}


def run_condition(scene_id, condition, calibration_scores, clear_scores, clear_features,
                  suffix_scores, suffix_features, *, method, probability_provider=None,
                  class_support_provider=None, input_artifact_sha256=None):
    """Run one fresh arm/controller over a common clear prefix and one render suffix.

    Inputs are already scored frame arrays. `probability_provider(t)` lazily supplies
    native-resolution 14×H×W probability maps when exact set-size outputs are wanted.
    This API does not load files or touch dataset pixels itself.
    """
    if condition not in CONDITIONS or method not in METHODS:
        raise ReplayProtocolError('unknown registered render condition or method')
    clear_s = _validate_vector(clear_scores, 'clear scores')
    suffix_s = _validate_vector(suffix_scores, 'suffix scores')
    clear_f = _validate_features(clear_features, 'clear features')
    suffix_f = _validate_features(suffix_features, 'suffix features')
    cal = np.asarray(calibration_scores, dtype=np.float64)
    anchor = calibration_anchor(cal, ALPHA)['anchor']
    settings = {'method': method, 'alpha': ALPHA, 'anchor': anchor, 'latency': 1, 'sparsity': 1,
                'risk_window': RISK_WINDOW, 'recovery_risk': RECOVERY_RISK, 'horizon': 30,
                'eta': .02, 'ki': .05, 'csat': 2., 'anti_windup': method in ('pi', 'history_pid', 'feature_pid'),
                'rolling_window': 10, 'rolling_minimum': 10, 'scorecaster_minimum': 10,
                'scorecaster_window': 80, 'scorecaster_ridge': .1, 'scorecaster_clamp': .25,
                'source_sha256': SOURCE_SHA256, 'runtime_identity': RUNTIME_IDENTITY,
                'calibration_score_sha256': hashlib.sha256(cal.astype('<f8', copy=False).tobytes()).hexdigest(),
                'revision': 'vkitti-clear-prefix-adverse-suffix-v1'}
    kwargs = {'anchor': anchor, 'alpha': ALPHA, 'method': method, 'eta': .02,
              'ki': .05, 'csat': 2., 'anti_windup': settings['anti_windup'],
              'latency': 1, 'sparse': 1, 'calibration_scores': cal if method == 'aci' else None,
              'window': 10, 'scorecaster_min': 10, 'scorecaster_window': 80,
              'ridge': .1, 'scorecaster_clamp': .25}
    controller = CausalController(**kwargs)
    replay = FeedbackReplay(controller, scope='evaluation')
    all_scores = np.r_[clear_s, suffix_s]
    all_features = np.vstack([clear_f, suffix_f])
    thresholds = replay.run(all_scores, all_features, alpha=ALPHA, risk_window=RISK_WINDOW,
                            recovery_risk=RECOVERY_RISK, horizon=30)
    if thresholds.shape != (PREFIX + SUFFIX,) or not np.isfinite(thresholds).all():
        raise ReplayProtocolError('replay returned malformed threshold trajectory')
    at_switch = [event for event in controller.received_feedback if event[0] == PREFIX]
    if at_switch != [(PREFIX, PREFIX - 1, float(clear_s[-1]))]:
        raise ReplayProtocolError('at the switch only clear-prefix frame 29 may be released')
    suffix_events = [(decision, index) for decision, index, _ in controller.received_feedback if decision > PREFIX]
    first_adverse_release = min((d for d, i in suffix_events if i >= PREFIX), default=None)
    if first_adverse_release != PREFIX + 1:
        raise ReplayProtocolError('first adverse outcome must be delayed until the next decision')
    metrics = _metrics(suffix_s, thresholds[PREFIX:], abs(float(thresholds[PREFIX] - thresholds[PREFIX - 1])))
    metrics.update(_set_metrics(thresholds[PREFIX:], probability_provider))
    if class_support_provider is None:
        metrics['guardrail_pixel_support'] = None
        metrics['guardrail_top1_error_count'] = None
    else:
        guard_support, guard_errors = 0, 0
        for i in range(PREFIX, PREFIX + SUFFIX):
            value = class_support_provider(i)
            guard_support += int(value['guardrail_pixel_count'])
            guard_errors += int(value['guardrail_top1_error_count'])
        metrics['guardrail_pixel_support'] = guard_support
        metrics['guardrail_top1_error_count'] = guard_errors
    return {'scene_id': str(scene_id), 'condition': condition, 'method': method,
            'settings': settings, 'settings_sha256': _digest(settings),
            'input_artifact_sha256': input_artifact_sha256,
            'input_sha256': hashlib.sha256(
                str(scene_id).encode() + condition.encode() +
                str(input_artifact_sha256 or '').encode() +
                all_scores.astype('<f8', copy=False).tobytes() +
                all_features.astype('<f8', copy=False).tobytes()).hexdigest(),
            'thresholds': thresholds.tolist(), 'clear_prefix_thresholds': thresholds[:PREFIX].tolist(),
            'post_switch_thresholds': thresholds[PREFIX:].tolist(),
            'feedback_released_at_switch': [{'decision_index': d, 'frame_index': i}
                                            for d, i, _ in at_switch],
            'first_adverse_feedback_decision_index': first_adverse_release,
            'feedback_event_count': len(controller.released), 'metrics': metrics}


def run_scene(scene_id, calibration_scores, clear_scores, clear_features, condition_suffixes):
    """Run all seven arms for clone/fog/rain and enforce exact clear-prefix parity."""
    if str(scene_id) not in ('02', '20'):
        raise ReplayProtocolError('replay runner accepts only registered held-out scene IDs 02/20')
    if tuple(sorted(condition_suffixes)) != tuple(sorted(CONDITIONS)):
        raise ReplayProtocolError('exactly clone, fog, and rain suffixes are required')
    rows = []
    for method in METHODS:
        arm_rows = []
        for condition in CONDITIONS:
            data = condition_suffixes[condition]
            arm_rows.append(run_condition(scene_id, condition, calibration_scores, clear_scores,
                                          clear_features, data['scores'], data['features'], method=method,
                                          probability_provider=data.get('probability_provider'),
                                          class_support_provider=data.get('class_support_provider'),
                                          input_artifact_sha256=data.get('input_artifact_sha256')))
        ref = np.asarray(arm_rows[0]['clear_prefix_thresholds'], dtype=np.float64)
        if not all(np.array_equal(ref, np.asarray(row['clear_prefix_thresholds'])) for row in arm_rows[1:]):
            raise ReplayProtocolError(f'clear-prefix controller trajectories differ for arm {method}')
        rows.extend(arm_rows)
    return {'scene_id': str(scene_id), 'condition_count': 3, 'method_count': len(METHODS),
            'clear_prefix_trajectory_match_by_method': True, 'arm_results': rows}


def summarize_two_scenes(scene_results):
    """Equal-weight descriptive arm summaries for the two registered held-out scenes."""
    if len(scene_results) != 2 or {str(r['scene_id']) for r in scene_results} != {'02', '20'}:
        raise ReplayProtocolError('scene summaries must contain exactly held-out scenes 02 and 20')
    rows = [arm for result in scene_results for arm in result['arm_results']]
    grouped = {}
    for row in rows:
        grouped.setdefault((row['condition'], row['method']), []).append(row)
    summaries = []
    scalar_metrics = ('post_switch_breach_rate', 'post_switch_risk_excess',
                     'first_crossing_post_step', 'restricted_log_endpoint',
                     'normalized_area_above_risk_010',
                     'within_suffix_mean_absolute_threshold_change',
                     'switch_boundary_threshold_jump', 'native_mean_set_size',
                     'native_fraction_multi_class')
    for (condition, method), arm_rows in sorted(grouped.items()):
        if len(arm_rows) != 2:
            raise ReplayProtocolError('each condition/method must have one result per scene')
        summary = {'condition': condition, 'method': method, 'scene_count': 2,
                   'scene_ids': sorted(row['scene_id'] for row in arm_rows),
                   'aggregation': 'equal-weight mean across two scenes; descriptive only',
                   'recovery_hit_scene_count': sum(not row['metrics']['recovery_censored'] for row in arm_rows),
                   'recovery_censored_scene_count': sum(row['metrics']['recovery_censored'] for row in arm_rows),
                   'raw_first_crossing_post_steps': [row['metrics']['first_crossing_post_step'] for row in arm_rows],
                   'rolling_risk_equal_scene_mean': np.mean(
                       [row['metrics']['post_switch_rolling_risk'] for row in arm_rows], axis=0).tolist()}
        for name in scalar_metrics:
            values = [row['metrics'][name] for row in arm_rows]
            summary[name] = None if any(v is None for v in values) else float(np.mean(values))
        summary['guardrail_pixel_support_total'] = (
            None if any(row['metrics']['guardrail_pixel_support'] is None for row in arm_rows) else
            int(sum(row['metrics']['guardrail_pixel_support'] for row in arm_rows)))
        summary['guardrail_top1_error_count_total'] = (
            None if any(row['metrics']['guardrail_top1_error_count'] is None for row in arm_rows) else
            int(sum(row['metrics']['guardrail_top1_error_count'] for row in arm_rows)))
        summaries.append(summary)
    if len(summaries) != len(CONDITIONS) * len(METHODS):
        raise ReplayProtocolError('two-scene summary lacks registered condition/method arms')
    return {'scene_count': 2, 'condition_count': 3, 'method_count': len(METHODS),
            'inference_scope': 'two-scene descriptive means; no significance test',
            'arm_summaries': summaries}

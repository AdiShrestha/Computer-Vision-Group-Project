#!/usr/bin/env python3
"""Synthetic D1 replay. Decision export is separate from private evaluator metrics."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from core.confseg_core import CausalController, FeedbackReplay, crc_threshold, paired_calibration_draw

METHODS = ('static', 'aci', 'qt', 'rolling', 'pi', 'history_pid', 'feature_pid')
ALGORITHM_REVISION = 'confseg-controlled-weather-v1'

def _runtime_identity():
    packages = {}
    for package in ('numpy', 'pyarrow', 'scikit-learn'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = 'missing'
    return {'python': platform.python_version(), 'implementation': platform.python_implementation(),
            'platform': platform.platform(), 'packages': packages}

def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def _mean(values):
    clean = [float(x) for x in values if x is not None]
    return float(np.mean(clean)) if clean else None

def main():
    p = argparse.ArgumentParser()
    p.add_argument('input')
    p.add_argument('out')
    args = p.parse_args()
    source = Path(args.input)
    data = json.loads(source.read_text())
    if data.get('synthetic_fixture') is not True:
        raise ValueError('this milestone replay runner accepts only marked synthetic fixtures')
    episodes = data.get('episodes') or [data]
    pool = np.asarray(data['calibration_pool'], dtype=np.float64)
    seeds = [int(s) for s in data.get('seeds', [data.get('seed', 101)])]
    n_cal = int(data.get('calibration_n', min(250, len(pool))))
    alpha = float(data.get('alpha_risk', data.get('alpha', .10)))
    latency, sparse = int(data.get('latency', 1)), int(data.get('sparse', 1))
    window = int(data.get('risk_window', 10))
    recovery_risk = float(data.get('recovery_risk', .15))
    horizon = int(data.get('horizon', 30))
    scorecaster_ridge = float(data.get('scorecaster_ridge', .1))
    scorecaster_clamp = float(data.get('scorecaster_clamp', .25))
    runtime_identity = _runtime_identity()
    if not seeds or not 0 < alpha < 1 or window < 1:
        raise ValueError('invalid replay settings')
    decisions, episode_metrics, arm_settings = [], [], {}
    draw_digests = {}

    for seed in seeds:
        calibration = paired_calibration_draw(pool, seed, n_cal)
        draw_digests[str(seed)] = hashlib.sha256(calibration.tobytes()).hexdigest()
        anchor = crc_threshold(calibration, alpha)
        for method in METHODS:
            cfg = {
                'method': method, 'alpha_risk': alpha, 'anchor': anchor,
                'latency': latency, 'sparse': sparse, 'risk_window': window,
                'calibration_sha256': draw_digests[str(seed)],
                'eta': float(data.get('eta', .02)), 'ki': float(data.get('ki', .05)),
                'csat': float(data.get('csat', 2.)), 'anti_windup': True,
                'rolling_window': int(data.get('rolling_window', 10)),
                'scorecaster_min': int(data.get('scorecaster_min', 10)),
                'scorecaster_window': int(data.get('scorecaster_window', 80)),
                'scorecaster_ridge': scorecaster_ridge,
                'scorecaster_clamp': scorecaster_clamp,
                'recovery_risk': recovery_risk, 'horizon': horizon,
                'algorithm_revision': ALGORITHM_REVISION,
                'runtime_identity': runtime_identity,
            }
            arm_settings[f'{method}/{seed}'] = dict(cfg, sha256=_digest(cfg))
            for ep in episodes:
                frame_index = ep.get('frame_index', list(range(len(ep['score_tau05']))))
                n = len(ep['score_tau05'])
                if frame_index != list(range(n)):
                    raise ValueError('frame_index must be episode-local and monotonic')
                kwargs = dict(anchor=anchor, alpha=alpha, method=method,
                              eta=cfg['eta'], ki=cfg['ki'], csat=cfg['csat'],
                              anti_windup=True, latency=latency, sparse=sparse,
                              calibration_scores=calibration if method == 'aci' else None,
                              window=cfg['rolling_window'], scorecaster_min=cfg['scorecaster_min'],
                              scorecaster_window=cfg['scorecaster_window'], ridge=cfg['scorecaster_ridge'],
                              scorecaster_clamp=cfg['scorecaster_clamp'])
                ctl = CausalController(**kwargs)
                evaluator = FeedbackReplay(ctl, scope=ep.get('scope', 'evaluation'))
                thresholds = evaluator.run(ep['score_tau05'], ep.get('features'), ep.get('efficiency'), alpha=alpha, risk_window=window, recovery_risk=recovery_risk, horizon=horizon)
                met = evaluator.private_metrics
                met.update({'seed': seed, 'method': method,
                            'episode_id': str(ep['episode_id']), 'cluster_id': str(ep['cluster_id']),
                            'condition': str(ep.get('condition', 'unspecified')),
                            'feedback_events_released': len(ctl.released),
                            'eligible_feedback_events': sum(i % sparse == 0 for i in range(max(0, n-latency))),
                            'settings_sha256': arm_settings[f'{method}/{seed}']['sha256']})
                episode_metrics.append(met)
                for t, threshold in enumerate(thresholds):
                    decisions.append({'seed': seed, 'method': method,
                                      'episode_id': str(ep['episode_id']), 'cluster_id': str(ep['cluster_id']),
                                      'condition': str(ep.get('condition', 'unspecified')),
                                      'decision_index': t, 'threshold': float(threshold),
                                      'settings_sha256': arm_settings[f'{method}/{seed}']['sha256']})

    decisions.sort(key=lambda r: (r['seed'], r['method'], r['cluster_id'], r['episode_id'], r['decision_index']))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ['seed', 'method', 'episode_id', 'cluster_id', 'condition', 'decision_index', 'threshold', 'settings_sha256']
    with out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(decisions)

    # Episode constituents are retained. Episodes first average within their physical cluster;
    # clusters then contribute equally to the seed-level descriptive rollup.
    metric_names = ('risk', 'risk_excess', 'excess_risk_area', 'restricted_log_endpoint', 'threshold_tv')
    by_cluster = {}
    for item in episode_metrics:
        key = (item['seed'], item['method'], item['cluster_id'])
        by_cluster.setdefault(key, []).append(item)
    cluster_rollup = []
    for (seed, method, cluster), items in sorted(by_cluster.items()):
        row = {'seed': seed, 'method': method, 'cluster_id': cluster,
               'episode_count': len(items), 'episode_ids': sorted(i['episode_id'] for i in items)}
        for name in metric_names:
            row[name] = _mean([i[name] for i in items])
        cluster_rollup.append(row)
    equal_cluster = []
    for seed in seeds:
        for method in METHODS:
            rows = [r for r in cluster_rollup if r['seed'] == seed and r['method'] == method]
            equal_cluster.append({'seed': seed, 'method': method, 'cluster_count': len(rows),
                                  **{name: _mean([r[name] for r in rows]) for name in metric_names}})
    private = {
        'synthetic_only': bool(data.get('synthetic_fixture', False)),
        'data_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'resolved_settings_by_arm_seed': arm_settings,
        'paired_calibration_draw_sha256_by_seed': draw_digests,
        'episode_metrics': episode_metrics,
        'cluster_rollup_equal_episode_weight_within_cluster': cluster_rollup,
        'equal_cluster_rollup_by_seed': equal_cluster,
        'terminal_status': 'synthetic_complete',
    }
    out.with_suffix('.private.json').write_text(json.dumps(private, indent=2, allow_nan=False) + '\n')

if __name__ == '__main__':
    main()

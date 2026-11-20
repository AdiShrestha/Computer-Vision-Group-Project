"""Causal scalar controllers with a hard delayed-feedback ingestion boundary."""
import math
import numpy as np

class CausalController:
    def __init__(self, anchor, alpha=.10, method='pid', eta=.02, ki=.05, csat=2.,
                 anti_windup=True, latency=1, sparse=1, calibration_scores=None,
                 window=10, scorecaster_min=10, scorecaster_window=80, ridge=.1,
                 scorecaster_clamp=.25):
        if not 0 <= anchor <= 1 or not 0 < alpha < 1 or latency < 1 or sparse < 1:
            raise ValueError('invalid controller settings')
        self.cal = None if calibration_scores is None else np.asarray(calibration_scores, dtype=np.float64)
        if method == 'aci' and (self.cal is None or self.cal.ndim != 1 or not self.cal.size or
                                not np.isfinite(self.cal).all() or np.any((self.cal < 0) | (self.cal > 1))):
            raise ValueError('ACI requires a finite paired calibration score vector in [0,1]')
        self.anchor, self.alpha, self.method = float(anchor), float(alpha), method
        self.eta, self.ki, self.csat = float(eta), float(ki), float(csat)
        self.anti_windup, self.L, self.q = bool(anti_windup), int(latency), int(sparse)
        self.window, self.scorecaster_min = int(window), int(scorecaster_min)
        self.scorecaster_window, self.ridge = int(scorecaster_window), float(ridge)
        self.scorecaster_clamp = float(scorecaster_clamp)
        if self.ridge < 0 or not math.isfinite(self.ridge) or self.scorecaster_clamp <= 0 or not math.isfinite(self.scorecaster_clamp):
            raise ValueError('invalid scorecaster ridge/clamp settings')
        self.t = -1
        self.raw, self.thresholds, self.errors, self.released = {}, [], {}, set()
        self.last_feedback_index = -1
        self.features_by_t, self.pairs = {}, []
        self.aci_alpha = float(alpha)
        self.received_feedback = []

    def release_feedback(self, index, score, decision_index):
        """Ingest a single evaluator-released score immediately before its eligible decision."""
        if decision_index != self.t + 1:
            raise ValueError('feedback must be released immediately before the next decision')
        if index not in self.raw or index >= decision_index:
            raise ValueError('feedback for undecided or current frame')
        if index > decision_index - self.L or index % self.q:
            raise ValueError('feedback is early or outside the sparsity schedule')
        if index in self.released or index <= self.last_feedback_index:
            raise ValueError('duplicate or out-of-order feedback')
        score = float(score)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('feedback score must be finite in [0,1]')
        error = float(score > self.thresholds[index])
        g = error - self.alpha
        self.errors[index] = g
        self.mark_feedback_score(index, score)
        self.released.add(index)
        self.last_feedback_index = index
        self.received_feedback.append((decision_index, index, score))
        if index in self.features_by_t:
            self.pairs.append((self.features_by_t[index].copy(), score))
        if self.method == 'aci':
            # Original ACI sign convention; alpha is deliberately not projected.
            self.aci_alpha += self.eta * (self.alpha - error)

    @staticmethod
    def outward_windup(raw_threshold, excess_error):
        return (raw_threshold >= 1. and excess_error > 0.) or (raw_threshold <= 0. and excess_error < 0.)

    @staticmethod
    def _rank_threshold(scores, alpha):
        vals = np.sort(np.asarray(scores, dtype=np.float64))
        k = int(np.ceil((len(vals) + 1) * (1. - alpha) - 1e-12))
        return 1. if k > len(vals) else float(vals[k - 1])

    def _scorecaster_correction(self, current, mode):
        if mode == 'none' or len(self.pairs) < self.scorecaster_min:
            return 0.
        pairs = self.pairs[-self.scorecaster_window:]
        ys = np.asarray([p[1] for p in pairs], dtype=np.float64)
        if mode == 'history':
            Xc = np.ones((len(pairs), 1), dtype=np.float64)
            z = np.ones(1, dtype=np.float64)
        elif mode == 'features':
            Z = np.stack([p[0] for p in pairs])
            mu = Z.mean(axis=0)
            centered = Z - mu
            Xc = np.column_stack([np.ones(len(Z)), centered])
            z = np.r_[1., np.asarray(current, dtype=np.float64) - mu]
        else:
            raise ValueError('unknown scorecaster mode')
        penalty = np.eye(Xc.shape[1], dtype=np.float64) * self.ridge
        penalty[0, 0] = 0.
        beta = np.linalg.solve(Xc.T @ Xc + penalty, Xc.T @ ys)
        residual = ys - Xc @ beta
        k = int(np.ceil((len(residual) + 1) * (1. - self.alpha) - 1e-12))
        if k > len(residual):
            return 0.  # registered static-anchor fallback when residual rank is unavailable
        qres = float(np.sort(residual)[k - 1])
        return float(np.clip(z @ beta + qres - self.anchor,
                             -self.scorecaster_clamp, self.scorecaster_clamp))

    def step(self, t, features=None):
        if t != self.t + 1:
            raise ValueError('decision indices must increase by one')
        if features is None:
            z = np.empty(0, dtype=np.float64)
        else:
            z = np.asarray(features, dtype=np.float64)
            if z.ndim != 1 or not np.isfinite(z).all():
                raise ValueError('current features must be a finite vector')
            z = z.copy()
        self.t = t
        self.features_by_t[t] = z
        method = {'intercept_pid': 'history_pid', 'pid': 'feature_pid'}.get(self.method, self.method)
        if method == 'static':
            raw = self.anchor
        elif method == 'aci':
            if self.aci_alpha <= 0.:
                raw = 1.
            elif self.aci_alpha >= 1.:
                raw = 0.  # LAC top-1 fallback remains active at lambda=0
            else:
                raw = self._rank_threshold(self.cal, self.aci_alpha)
        elif method == 'qt':
            raw = self.anchor + self.eta * sum(self.errors.values())
        elif method == 'rolling':
            recent = [self._score_for_error_idx(i) for i in sorted(self.errors)[-self.window:]]
            if len(recent) < self.window:
                raw = self.anchor
            else:
                k = int(np.ceil((len(recent) + 1) * (1. - self.alpha) - 1e-12))
                raw = 1. if k > len(recent) else float(np.sort(recent)[k - 1])
        elif method in ('pi', 'history_pid', 'feature_pid'):
            accepted = [g for i, g in sorted(self.errors.items())
                        if not (self.anti_windup and self.outward_windup(self.raw[i], g))]
            x = float(sum(accepted))
            arg = x * math.log(t + 1.) / ((t + 1.) * self.csat)
            if abs(arg) >= math.pi / 2.:
                integ = self.ki * math.copysign(1e6, arg)
            else:
                integ = self.ki * math.tan(arg)
            latest = self.errors[max(self.errors)] if self.errors else 0.
            prop = self.eta * latest
            mode = 'none' if method == 'pi' else ('history' if method == 'history_pid' else 'features')
            corr = self._scorecaster_correction(z, mode)
            raw = self.anchor + prop + integ + corr
        else:
            raise ValueError(f'unknown controller method: {method}')
        self.raw[t] = float(raw)
        threshold = float(np.clip(raw, 0., 1.))
        self.thresholds.append(threshold)
        return threshold

    def _score_for_error_idx(self, index):
        # Rolling only needs scores, which the feedback ledger supplies at release.
        return self.score_by_index[index]

    def mark_feedback_score(self, index, score):
        """Called only by the feedback boundary as part of release_feedback."""
        if not hasattr(self, 'score_by_index'):
            self.score_by_index = {}
        if index in self.score_by_index:
            raise ValueError('duplicate feedback score')
        self.score_by_index[index] = float(score)


class FeedbackReplay:
    """Private evaluator owns outcomes and releases them only at their due decision."""
    def __init__(self, controller, scope='evaluation'):
        self.controller, self.scope = controller, scope
        self.used, self.trace = set(), []
        self.private_metrics = None

    def run(self, scores, features=None, efficiency=None, alpha=.10, risk_window=10, recovery_risk=.15, horizon=30, stop_after=None):
        vals = np.asarray(scores, dtype=np.float64)
        if vals.ndim != 1:
            raise ValueError('evaluator scores must be a vector')
        if self.scope not in ('selection', 'evaluation'):
            raise ValueError('replay scope must be selection or evaluation')
        if features is not None and len(features) != len(vals):
            raise ValueError('feature/outcome alignment mismatch')
        if efficiency is not None and len(efficiency.get('grid', [])) == 0:
            raise ValueError('efficiency grid cannot be empty')
        thresholds = []
        steps = len(vals) if stop_after is None else min(len(vals), int(stop_after))
        if steps < 0:
            raise ValueError('stop_after must be nonnegative')
        for t in range(steps):
            due = t - self.controller.L
            # At most one frame can newly become eligible per decision; q gates label cadence.
            if due >= 0 and due % self.controller.q == 0 and due not in self.used:
                self.controller.release_feedback(due, vals[due], t)
                self.used.add(due)
            z = None if features is None else features[t]
            threshold = self.controller.step(t, z)
            thresholds.append(threshold)
            self.trace.append((t, threshold))
        self.private_metrics = _evaluate(vals[:steps], np.asarray(thresholds), efficiency, window=risk_window, alpha=alpha, recovery_risk=recovery_risk, horizon=horizon)
        return np.asarray(thresholds)


def _evaluate(scores, thresholds, efficiency=None, window=10, alpha=.10, recovery_risk=.15, horizon=30):
    errors = (scores > thresholds).astype(float)
    rolling = np.asarray([errors[max(0, t - window + 1):t + 1].mean() for t in range(len(errors))])
    hit = next((t for t in range(window - 1, len(rolling)) if rolling[t] <= recovery_risk), None)
    first_step = None if hit is None else hit + 1
    restricted = min(first_step if first_step is not None else horizon, horizon)
    metrics = {
        'risk': float(errors.mean()) if len(errors) else None,
        'risk_excess': float(max(0., errors.mean() - alpha)) if len(errors) else None,
        'breach': errors.astype(int).tolist(),
        'rolling_risk': rolling.tolist(),
        'excess_risk_area': float(np.maximum(0., rolling - alpha).mean()) if len(errors) else None,
        'first_crossing_step': first_step,
        'first_crossing_decision_index': hit,
        'censored': hit is None,
        'restricted_log_endpoint': float(np.log1p(restricted)),
        'threshold_tv': float(np.abs(np.diff(thresholds)).mean()) if len(thresholds) > 1 else 0.,
        'AR': None, 'APF': None,
    }
    if efficiency is not None:
        grid = np.asarray(efficiency['grid'], dtype=np.float64)
        ars, apfs = [], []
        for i, threshold in enumerate(thresholds):
            ars.append(float(np.interp(threshold, grid, np.asarray(efficiency['AR'][i], dtype=float))))
            apfs.append(float(np.interp(threshold, grid, np.asarray(efficiency['APF'][i], dtype=float))))
        metrics['AR'] = ars
        metrics['APF'] = apfs
    return metrics

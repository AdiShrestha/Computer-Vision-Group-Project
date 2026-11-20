import unittest

import numpy as np
import torch

from core.confseg_core.vkitti import (calibration_anchor, calibration_rank, frame_score,
                                      restore_native_probabilities, set_size_statistics)
from replay.vkitti_replay import (CONDITIONS, METHODS, PREFIX, ReplayProtocolError,
                                  run_condition, run_scene, summarize_two_scenes)
from stage0.vkitti_score import ScoreError, strict_state_arrays


class VKittiScoring(unittest.TestCase):
    def test_tau05_rank_ignore_guardrail_and_undefined_distinction(self):
        p = np.full((14, 1, 4), .0, dtype=np.float64)
        p[6, 0, 0] = .8; p[1, 0, 0] = .2
        p[0, 0, 1] = .4; p[1, 0, 1] = .6
        p[0, 0, 2] = .8; p[1, 0, 2] = .2
        p[1, 0, 3] = .7; p[2, 0, 3] = .3
        y = np.array([[6, 255, 0, 2]], dtype=np.uint8)
        result = frame_score(p, y)
        # Three valid pixels: score candidates are 0, 0, and 0.7; k=3.
        self.assertAlmostEqual(result['score_tau05'], .7)
        self.assertEqual(result['valid_pixel_count'], 3)
        self.assertEqual(result['native_class_pixel_counts'][6], 1)  # GuardRail is scored.
        self.assertEqual(result['native_class_pixel_counts'].sum(), 3)  # Undefined/255 is ignored.
        self.assertEqual(result['guardrail_pixel_count'], 1)
        self.assertEqual(result['guardrail_top1_error_count'], 0)
        self.assertEqual(result['top1_error_counts_by_class'][2], 1)

    def test_calibration_rank_ties_and_endpoint(self):
        self.assertEqual(calibration_rank(339, .1), 306)
        scores = np.array([.25] * 340)
        self.assertEqual(calibration_anchor(scores, .1)['anchor'], .25)
        self.assertEqual(calibration_anchor(scores, .1)['rank_one_based'], 307)

    def test_all_void_and_invalid_labels_block(self):
        p = np.full((14, 1, 1), 1. / 14)
        with self.assertRaisesRegex(ValueError, 'all-void'):
            frame_score(p, np.array([[255]], dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, 'label outside'):
            frame_score(p, np.array([[14]], dtype=np.uint8))

    def test_crop_before_restore_and_native_softmax(self):
        # Native 2x2 is embedded in a 4x4 crop with unrelated extreme padding logits.
        logits = torch.full((14, 160, 512), -100.)
        logits[0, 0:2, 0:2] = torch.tensor([[4., 3.], [2., 1.]])
        logits[1, 0:2, 0:2] = 0.
        logits[2, 100:, 100:] = 1000.
        geometry = {'pad_top': 0, 'pad_left': 0, 'scaled_height': 2, 'scaled_width': 2,
                    'native_height': 2, 'native_width': 2}
        actual = restore_native_probabilities(logits, geometry)
        crop = logits[..., :2, :2].unsqueeze(0)
        expected = torch.softmax(torch.nn.functional.interpolate(crop, size=(2, 2), mode='bilinear',
                                      align_corners=False), dim=1)[0].numpy()
        self.assertTrue(np.array_equal(actual, expected))
        self.assertTrue(np.allclose(actual.sum(axis=0), 1.))

    def test_set_size_top1_fallback_and_exact_threshold(self):
        p = np.zeros((14, 1, 2), dtype=np.float32)
        p[2, 0, 0] = .8; p[3, 0, 0] = .2
        p[0, 0, 1] = .5; p[1, 0, 1] = .5
        zero = set_size_statistics(p, 0.)
        self.assertEqual(zero['mean_set_size'], 1.)
        self.assertEqual(zero['fraction_multi_class'], 0.)
        exact = set_size_statistics(p, .5)
        self.assertEqual(exact['mean_set_size'], 1.5)
        self.assertEqual(exact['fraction_multi_class'], .5)

    def test_float32_activation_boundary_is_compared_in_float64(self):
        p = np.array([[[.9]], [[.1]]], dtype=np.float32)
        activation = float(1.0 - np.float64(p[1, 0, 0]))
        below = np.nextafter(np.float64(activation), -np.inf)
        just_below = set_size_statistics(p, below)
        exactly_at = set_size_statistics(p, activation)
        self.assertEqual(just_below['mean_set_size'], 1.)
        self.assertEqual(just_below['fraction_multi_class'], 0.)
        self.assertEqual(exactly_at['mean_set_size'], 2.)
        self.assertEqual(exactly_at['fraction_multi_class'], 1.)
        self.assertEqual(set_size_statistics(p, 0.)['mean_set_size'], 1.)

    def test_checkpoint_tensor_validation_rejects_mismatch(self):
        model = torch.nn.Linear(2, 3)
        expected = model.state_dict()
        arrays = {name: value.detach().numpy().copy() for name, value in expected.items()}
        self.assertEqual(set(strict_state_arrays(expected, arrays)), set(expected))
        bad_shape = dict(arrays); bad_shape['weight'] = np.zeros((2, 3), dtype=np.float32)
        with self.assertRaisesRegex(ScoreError, 'shape/dtype mismatch'):
            strict_state_arrays(expected, bad_shape)
        with self.assertRaisesRegex(ScoreError, 'tensor names'):
            strict_state_arrays(expected, {'weight': arrays['weight']})

    def test_clear_prefix_replay_switch_feedback_and_fresh_condition_trajectories(self):
        calibration = np.linspace(.01, .99, 339)
        clear_scores = np.full(PREFIX, .2)
        clear_features = np.tile([.4, .1, .8], (PREFIX, 1))
        suffixes = {condition: {'scores': np.full(PREFIX, value),
                               'features': np.tile([.5, .2, .7], (PREFIX, 1))}
                    for condition, value in zip(CONDITIONS, (.2, .8, .7))}
        result = run_scene('02', calibration, clear_scores, clear_features, suffixes)
        self.assertEqual(result['method_count'], len(METHODS))
        self.assertTrue(result['clear_prefix_trajectory_match_by_method'])
        self.assertEqual(len(result['arm_results']), len(METHODS) * 3)
        second = run_scene('20', calibration, clear_scores, clear_features, suffixes)
        summary = summarize_two_scenes([result, second])
        self.assertEqual(summary['scene_count'], 2)
        self.assertEqual(len(summary['arm_summaries']), len(METHODS) * 3)
        with self.assertRaisesRegex(ReplayProtocolError, 'held-out scene IDs'):
            run_scene('18', calibration, clear_scores, clear_features, suffixes)
        for arm in result['arm_results']:
            self.assertEqual(arm['feedback_released_at_switch'], [{'decision_index': 30, 'frame_index': 29}])
            self.assertEqual(arm['first_adverse_feedback_decision_index'], 31)
            self.assertEqual(len(arm['post_switch_thresholds']), 30)
            self.assertEqual(len(arm['metrics']['post_switch_rolling_risk']), 30)
            self.assertIn('switch_boundary_threshold_jump', arm['metrics'])

    def test_suffix_outcome_cannot_change_switch_decision_and_set_outputs(self):
        calibration = np.linspace(.01, .99, 339)
        clear_scores = np.full(30, .2)
        clear_features = np.tile([.2, .1, .9], (30, 1))
        suffix_features = np.tile([.4, .3, .8], (30, 1))
        one = np.zeros((14, 1, 2), dtype=np.float32); one[0] = .8; one[1] = .2
        probabilities = {t: one.copy() for t in range(30, 60)}
        provider = lambda t: probabilities[t]
        a = run_condition('20', 'fog', calibration, clear_scores, clear_features,
                          np.full(30, .1), suffix_features, method='feature_pid', probability_provider=provider)
        b = run_condition('20', 'fog', calibration, clear_scores, clear_features,
                          np.r_[.99, np.full(29, .1)], suffix_features, method='feature_pid', probability_provider=provider)
        self.assertEqual(a['thresholds'][:31], b['thresholds'][:31])
        self.assertEqual(a['metrics']['native_mean_set_size'], 1.)
        self.assertEqual(a['metrics']['native_fraction_multi_class'], 0.)
        with self.assertRaisesRegex(ReplayProtocolError, 'exactly 30'):
            run_condition('20', 'rain', calibration, clear_scores[:-1], clear_features,
                          np.full(30, .2), suffix_features, method='static')


if __name__ == '__main__':
    unittest.main()

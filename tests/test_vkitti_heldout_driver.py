import unittest

from stage0.vkitti_heldout import _selection_keys, _synthetic_dry_run


class HeldoutDriverTests(unittest.TestCase):
    def test_frozen_selection_and_complete_synthetic_replay(self):
        keys = _selection_keys()
        self.assertEqual(len(keys), 240)
        self.assertEqual(len(set(keys)), 240)
        self.assertEqual(sum(condition == 'clone' for _, condition, _ in keys), 120)
        self.assertEqual(sum(condition == 'fog' for _, condition, _ in keys), 60)
        self.assertEqual(sum(condition == 'rain' for _, condition, _ in keys), 60)
        report = _synthetic_dry_run()
        self.assertEqual(report['status'], 'passed')
        self.assertFalse(report['dataset_pixels_opened'])
        self.assertEqual(report['scene_condition_arm_rows'], 42)
        self.assertEqual(report['two_scene_descriptive_summaries'], 21)
        self.assertEqual(report['switch_feedback'], 't=30 receives frame 29; first adverse release at t=31')
        self.assertEqual(report['all_arm_set_size_callbacks'], 'passed')


if __name__ == '__main__':
    unittest.main()

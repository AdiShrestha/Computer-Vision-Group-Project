import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from stage0.vkitti_census import CensusError, census

PALETTE = {
    'Terrain': (210, 0, 200), 'Sky': (90, 200, 255), 'Tree': (0, 199, 0),
    'Vegetation': (90, 240, 0), 'Building': (140, 140, 140), 'Road': (100, 60, 100),
    'GuardRail': (250, 100, 255), 'TrafficSign': (255, 255, 0), 'TrafficLight': (200, 200, 0),
    'Pole': (255, 130, 0), 'Misc': (80, 80, 80), 'Truck': (160, 60, 60),
    'Car': (255, 127, 80), 'Van': (0, 139, 139), 'Undefined': (0, 0, 0),
}
SCENES = ('01', '02', '06', '18', '20')
CONDITIONS = ('clone', 'fog', 'rain')


def add(tar, name, content):
    info = tarfile.TarInfo(name)
    info.size = len(content)
    tar.addfile(info, io.BytesIO(content))


def image_bytes(rgb, fmt):
    out = io.BytesIO()
    Image.fromarray(np.asarray([[rgb]], dtype=np.uint8), 'RGB').save(out, format=fmt)
    return out.getvalue()


class VKittiCensusTests(unittest.TestCase):
    def make_fixture(self, root, *, missing_mask=None, unknown_sample=False, duplicate=False):
        rgb_path, seg_path, meta_path = (root / n for n in ('rgb.tar', 'seg.tar', 'meta.tar.gz'))
        keys = [(s, c, f) for s in SCENES for c in CONDITIONS for f in range(3)]
        with tarfile.open(rgb_path, 'w') as tar:
            for s, c, f in keys:
                add(tar, f'fixture/Scene{s}/{c}/frames/rgb/Camera_0/rgb_{f:05}.jpg', image_bytes((1, 2, 3), 'JPEG'))
                if duplicate and (s, c, f) == ('01', 'clone', 0):
                    add(tar, f'fixture/Scene{s}/{c}/frames/rgb/Camera_0/rgb_{f:05}.jpg', image_bytes((1, 2, 3), 'JPEG'))
        with tarfile.open(seg_path, 'w') as tar:
            for s, c, f in keys:
                if (s, c, f) == missing_mask:
                    continue
                color = (123, 45, 67) if unknown_sample and (s, c, f) == ('01', 'clone', 0) else PALETTE['Road']
                add(tar, f'fixture/Scene{s}/{c}/frames/classSegmentation/Camera_0/classgt_{f:05}.png', image_bytes(color, 'PNG'))
        with tarfile.open(meta_path, 'w:gz') as tar:
            rows = ['Category r g b'] + [f'{k} {r} {g} {b}' for k, (r, g, b) in PALETTE.items()]
            add(tar, 'Scene01/clone/colors.txt', ('\n'.join(rows) + '\n').encode())
            for s in SCENES:
                for c in CONDITIONS:
                    contents = 'frame cameraID extra\n' + ''.join(f'{i} 0 x\n{i} 1 x\n' for i in range(3))
                    add(tar, f'Scene{s}/{c}/extrinsic.txt', contents.encode())
        return rgb_path, seg_path, meta_path

    def test_pairs_role_counts_and_decodes_only_development_mask(self):
        with tempfile.TemporaryDirectory() as d:
            paths = self.make_fixture(Path(d))
            report = census(*paths, strict_md5=False)
            self.assertEqual(report['total_matched'], 45)
            self.assertEqual(report['total_rgb_without_mask'], 0)
            self.assertEqual(report['scope_counts']['model_development']['clone']['matched'], 6)
            self.assertEqual(report['scope_counts']['held_out_evaluation']['fog']['matched'], 6)
            self.assertTrue(report['palette_check']['matches_publisher_palette'])
            self.assertEqual(report['palette_check']['sample_scene'], '01')
            self.assertFalse(report['protocol']['held_out_mask_pixels_decoded'])
            self.assertTrue(all(x['missing_pairs_from_metadata'] == 0 for x in report['per_scene_condition_counts']))
            serialized = json.dumps(report)
            self.assertNotIn('frames/rgb', serialized)
            self.assertNotIn('classgt_', serialized)

    def test_detects_missing_pair_and_bad_development_palette(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            paths = self.make_fixture(root, missing_mask=('01', 'clone', 2))
            report = census(*paths, strict_md5=False)
            self.assertEqual(report['total_rgb_without_mask'], 1)
            self.assertEqual(report['per_scene_condition_counts'][0]['missing_pairs_from_metadata'], 1)
            paths = self.make_fixture(root, unknown_sample=True)
            with self.assertRaisesRegex(CensusError, 'outside publisher palette'):
                census(*paths, strict_md5=False)

    def test_rejects_duplicate_member_key(self):
        with tempfile.TemporaryDirectory() as d:
            paths = self.make_fixture(Path(d), duplicate=True)
            with self.assertRaisesRegex(CensusError, 'repeated/conflicting'):
                census(*paths, strict_md5=False)


if __name__ == '__main__':
    unittest.main()

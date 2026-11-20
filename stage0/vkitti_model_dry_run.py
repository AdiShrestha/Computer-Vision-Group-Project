#!/usr/bin/env python3
"""CPU-only shape/finite dry run on one model-development RGB frame."""
import argparse
import io
import json
import platform
import tarfile

import numpy as np
from PIL import Image
import torch
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


def run(rgb_tar):
    torch.set_num_threads(2)
    with tarfile.open(rgb_tar, 'r:') as archive:
        member = next((m for m in archive if m.isfile() and
                       '/Scene01/clone/frames/rgb/Camera_0/' in '/' + m.name), None)
        if member is None:
            raise ValueError('no Scene01 clone camera-0 development RGB frame found')
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError('cannot read development RGB frame')
        with Image.open(io.BytesIO(stream.read())) as image:
            image = image.convert('RGB').resize((512, 256), Image.Resampling.BILINEAR)
            arr = np.asarray(image, dtype=np.float32) / 255.
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    model = deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None,
                                          num_classes=15).eval().cpu()
    with torch.inference_mode():
        logits = model(x)['out']
    if logits.shape != (1, 15, 256, 512) or not torch.isfinite(logits).all():
        raise RuntimeError('CPU model dry run returned invalid shape or values')
    return {'status': 'ok', 'scope': 'development RGB only; no labels or held-out pixels read',
            'architecture': 'torchvision DeepLabV3 MobileNetV3-Large', 'classes': 15,
            'initialization': 'random; weights=None and weights_backbone=None', 'device': 'cpu',
            'input_shape': list(x.shape), 'logits_shape': list(logits.shape),
            'finite_logits': True, 'python': platform.python_version(),
            'torch': torch.__version__, 'torchvision': __import__('torchvision').__version__,
            'mps_available': torch.backends.mps.is_available()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb-tar', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = run(args.rgb_tar)
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

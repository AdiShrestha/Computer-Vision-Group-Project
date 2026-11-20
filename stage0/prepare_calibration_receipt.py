"""Preserve the calibration-only receipt and record its two distinct scopes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    directory = parser.parse_args().directory
    current = directory / 'calibration_receipt.json'
    original = directory / 'calibration_receipt.original.json'
    correction = directory / 'calibration_receipt.correction.json'
    if original.exists() or correction.exists():
        raise SystemExit('Scope records already exist. Preserve the existing files.')
    raw = current.read_bytes()
    receipt = json.loads(raw)
    if (receipt.get('status'), receipt.get('phase'), receipt.get('scene'),
        receipt.get('condition'), receipt.get('frame_count')) != (
            'complete', 'calibration_only', '18', 'clone', 339):
        raise SystemExit('Expected the completed Scene18 clone calibration receipt.')
    scores_file = directory / 'scene18_clone_scores_tau05.npy'
    file_digest = hashlib.sha256(scores_file.read_bytes()).hexdigest()
    scores = np.load(scores_file, allow_pickle=False)
    vector_digest = hashlib.sha256(scores.astype('<f8', copy=False).tobytes()).hexdigest()
    if (scores.shape != (339,) or not np.isfinite(scores).all()
        or file_digest != receipt['score_file_sha256']
        or vector_digest != receipt['score_vector_sha256']
        or receipt['anchor_rank_one_based'] != 306
        or np.sort(scores)[305] != receipt['calibration_anchor']):
        raise SystemExit('Calibration receipt and score file disagree.')
    facts = {'scene18_clone_calibration_pixels_opened': True,
             'scene02_pixels_opened_during_calibration': False,
             'scene20_pixels_opened_during_calibration': False}
    detail = {'original_receipt_sha256': hashlib.sha256(raw).hexdigest(),
              'accepted_calibration_score_vector_sha256': vector_digest,
              'calibration_score_file_sha256': file_digest, 'scope_facts': facts}
    receipt.pop('calibration_or_held_out_pixels_opened', None)
    receipt.update(calibration_pixels_opened=True, held_out_pixels_opened=False)
    with original.open('xb') as file:
        file.write(raw)
    with correction.open('x', encoding='utf-8') as file:
        json.dump(detail, file, indent=2, sort_keys=True)
        file.write('\n')
    tmp = current.with_suffix('.tmp')
    with tmp.open('x', encoding='utf-8') as file:
        json.dump(receipt, file, indent=2, sort_keys=True)
        file.write('\n')
    tmp.replace(current)


if __name__ == '__main__':
    main()

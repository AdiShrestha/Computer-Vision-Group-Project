#!/usr/bin/env python3
"""Read-only Virtual KITTI 2 RGB/class-segmentation TAR pairing census."""
import argparse
import hashlib
import io
import json
import re
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath

SCENES = ('01', '02', '06', '18', '20')
ROLE = {'01': 'model_development', '06': 'model_development',
        '18': 'calibration', '02': 'held_out_evaluation', '20': 'held_out_evaluation'}
CONDITIONS = ('clone', 'fog', 'rain')
PALETTE_SCENE_CONDITION = ('01', 'clone')  # development only; never decode held-out label pixels
PUBLISHED_MD5 = {
    'rgb': '1e00a143a397c2c53aa9720a868fe34a',
    'class_segmentation': 'e8658ab49250e61f1caa174625563263',
}
MEMBER_RE = re.compile(r'^Scene(\d{2})/([^/]+)/frames/(rgb|classSegmentation)/Camera_(\d)/(rgb|classgt)_(\d+)\.(jpg|png)$')

class CensusError(ValueError):
    pass

def file_hashes(path):
    sha, md5, size = hashlib.sha256(), hashlib.md5(), 0
    with open(path, 'rb') as stream:
        while True:
            block = stream.read(4 << 20)
            if not block:
                break
            sha.update(block); md5.update(block); size += len(block)
    return {'bytes': size, 'sha256': sha.hexdigest(), 'md5': md5.hexdigest()}

def _normalize(name):
    parts = PurePosixPath(name).parts
    indices = [i for i, part in enumerate(parts) if re.fullmatch(r'Scene\d{2}', part)]
    if not indices:
        return None, None
    i = indices[0]
    return parts[:i], '/'.join(parts[i:])

def _scan_archive(path, kind):
    by_key = {}
    prefixes = set()
    total_members = recognized = malformed = 0
    dupe_count = 0
    try:
        with tarfile.open(path, mode='r|') as archive:
            for member in archive:
                if not member.isfile():
                    continue
                total_members += 1
                prefix, normalized = _normalize(member.name)
                if prefix is None:
                    if '/frames/' in member.name and ('/rgb/' in member.name or '/classsegmentation/' in member.name):
                        malformed += 1
                    continue
                prefixes.add(prefix)
                match = MEMBER_RE.fullmatch(normalized)
                if not match:
                    if '/frames/' in normalized and (kind == 'rgb' and '/rgb/' in normalized or kind == 'class_segmentation' and '/classSegmentation/' in normalized):
                        malformed += 1
                    continue
                scene, condition, datatype, camera, stem, frame, extension = match.groups()
                expected = 'rgb' if kind == 'rgb' else 'classSegmentation'
                valid_ext = 'jpg' if kind == 'rgb' else 'png'
                expected_stem = 'rgb' if kind == 'rgb' else 'classgt'
                if datatype != expected or extension != valid_ext or stem != expected_stem:
                    malformed += 1
                    continue
                recognized += 1
                key = (scene, condition, int(camera), int(frame))
                if key in by_key:
                    dupe_count += 1
                else:
                    by_key[key] = member.name
    except (OSError, tarfile.TarError) as exc:
        raise CensusError(f'cannot read {kind} TAR archive ({type(exc).__name__})') from exc
    if dupe_count:
        raise CensusError(f'{kind} TAR contains {dupe_count} repeated/conflicting frame keys')
    if len(prefixes) > 1:
        raise CensusError(f'{kind} TAR has inconsistent optional member prefixes ({len(prefixes)} variants)')
    if malformed:
        raise CensusError(f'{kind} TAR contains {malformed} malformed frame member paths')
    return by_key, {'regular_members': total_members, 'recognized_frame_members': recognized,
                    'optional_prefix_components': len(next(iter(prefixes))) if prefixes else 0}

def _parse_palette(text_tar):
    try:
        with tarfile.open(text_tar, mode='r|gz') as archive:
            for member in archive:
                if not member.isfile():
                    continue
                _, normalized = _normalize(member.name)
                if normalized == 'Scene01/clone/colors.txt':
                    data = archive.extractfile(member).read().decode('utf-8-sig')
                    break
            else:
                raise CensusError('text ground-truth archive lacks the development palette table')
    except (OSError, tarfile.TarError, UnicodeError) as exc:
        if isinstance(exc, CensusError):
            raise
        raise CensusError(f'cannot read publisher palette table ({type(exc).__name__})') from exc
    lines = [line.split() for line in data.splitlines() if line.strip()]
    if not lines or lines[0] != ['Category', 'r', 'g', 'b']:
        raise CensusError('publisher palette header is invalid')
    palette = {}
    for fields in lines[1:]:
        if len(fields) != 4:
            raise CensusError('publisher palette row is malformed')
        name = fields[0]
        try:
            rgb = tuple(map(int, fields[1:]))
        except ValueError as exc:
            raise CensusError('publisher palette has a non-integer channel') from exc
        if name in palette or any(v < 0 or v > 255 for v in rgb):
            raise CensusError('publisher palette has duplicate categories or invalid RGB values')
        palette[name] = rgb
    if len(set(palette.values())) != len(palette):
        raise CensusError('publisher palette has duplicate RGB values')
    if palette.get('Undefined') != (0, 0, 0):
        raise CensusError('publisher palette lacks the expected Undefined/black entry')
    return palette

def _decode_member(archive_path, member_name):
    try:
        with tarfile.open(archive_path, mode='r:') as archive:
            for member in archive:
                if member.isfile() and member.name == member_name:
                    stream = archive.extractfile(member)
                    if stream is None:
                        break
                    return stream.read()
    except (OSError, tarfile.TarError) as exc:
        raise CensusError(f'cannot read development sample mask ({type(exc).__name__})') from exc
    raise CensusError('development sample mask disappeared during palette check')

def _palette_check(seg_tar, sample_mask, palette):
    try:
        from PIL import Image
        import numpy as np
        with Image.open(io.BytesIO(_decode_member(seg_tar, sample_mask))) as image:
            rgb = np.asarray(image.convert('RGB'))
    except ImportError as exc:
        raise CensusError('Pillow is required for the in-memory RGB palette check') from exc
    actual = {tuple(map(int, color)) for color in np.unique(rgb.reshape(-1, 3), axis=0)}
    allowed = set(palette.values())
    unknown = len(actual - allowed)
    return {'sample_source_role': 'model_development', 'sample_condition': 'clone',
            'sample_scene': '01', 'decoded_image_mode': 'RGB',
            'width': int(rgb.shape[1]), 'height': int(rgb.shape[0]),
            'publisher_palette_categories': len(palette), 'publisher_palette_unique_rgb': len(allowed),
            'observed_unique_rgb_in_development_mask': len(actual),
            'unexpected_rgb_color_count': unknown, 'matches_publisher_palette': unknown == 0}

def _sequence_summary(keys, expected_indices=None):
    if not keys:
        return {'matched_frames': 0, 'eligible_30': False, 'longest_contiguous_run': 0,
                'native_index_gap_count': None, 'contiguous_from_zero': False}
    indices = sorted(k[3] for k in keys)
    gap_count = sum(max(0, b-a-1) for a, b in zip(indices, indices[1:]))
    longest, current = 1, 1
    for a, b in zip(indices, indices[1:]):
        current = current + 1 if b == a + 1 else 1
        longest = max(longest, current)
    if expected_indices is not None:
        gaps = len(set(expected_indices) - set(indices))
        contiguous = not gaps and indices == sorted(expected_indices)
    else:
        gaps = gap_count + indices[0]
        contiguous = indices[0] == 0 and gap_count == 0
    return {'matched_frames': len(indices), 'eligible_30': longest >= 30,
            'longest_contiguous_run': longest,
            'native_index_gap_count': gaps,
            'contiguous_from_zero': contiguous}

def _metadata_frame_keys(text_tar):
    """Read only camera/frame columns from publisher extrinsic metadata."""
    expected = {}
    try:
        with tarfile.open(text_tar, mode='r:gz') as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith('/extrinsic.txt'):
                    continue
                _, normalized = _normalize(member.name)
                parts = normalized.split('/') if normalized else []
                if len(parts) != 3 or not parts[0].startswith('Scene') or parts[0][5:] not in ROLE or parts[1] not in CONDITIONS:
                    continue
                frame_ids = set()
                stream = archive.extractfile(member)
                if stream is None:
                    raise CensusError('cannot read publisher camera metadata')
                header = stream.readline().decode('utf-8-sig').split()
                if header[:2] != ['frame', 'cameraID']:
                    raise CensusError('publisher camera metadata header is invalid')
                for raw in stream:
                    fields = raw.split()
                    if len(fields) < 2:
                        continue
                    try:
                        frame, camera = int(fields[0]), int(fields[1])
                    except ValueError as exc:
                        raise CensusError('publisher camera metadata has invalid frame keys') from exc
                    if camera == 0:
                        frame_ids.add(frame)
                expected[(parts[0][5:], parts[1])] = frame_ids
    except (OSError, tarfile.TarError, UnicodeError) as exc:
        raise CensusError(f'cannot read publisher camera metadata ({type(exc).__name__})') from exc
    missing = [(scene, cond) for scene in SCENES for cond in CONDITIONS if (scene, cond) not in expected]
    if missing or any(not expected[k] for k in expected):
        raise CensusError('publisher camera metadata is incomplete for the registered scenes/conditions')
    return expected

def census(rgb_tar, seg_tar, textgt_tar, *, strict_md5=True):
    rgb_path, seg_path = Path(rgb_tar), Path(seg_tar)
    if rgb_path.resolve() == seg_path.resolve():
        raise CensusError('RGB and segmentation inputs must be separate archives')
    rgb_keys, rgb_members = _scan_archive(rgb_path, 'rgb')
    seg_keys, seg_members = _scan_archive(seg_path, 'class_segmentation')
    rgb_meta, seg_meta = file_hashes(rgb_path), file_hashes(seg_path)
    rgb_meta['published_md5'] = PUBLISHED_MD5['rgb']
    seg_meta['published_md5'] = PUBLISHED_MD5['class_segmentation']
    rgb_meta['published_md5_matches'] = rgb_meta['md5'] == PUBLISHED_MD5['rgb']
    seg_meta['published_md5_matches'] = seg_meta['md5'] == PUBLISHED_MD5['class_segmentation']
    if strict_md5 and not rgb_meta['published_md5_matches']:
        raise CensusError('RGB archive MD5 does not match NAVER published digest')
    if strict_md5 and not seg_meta['published_md5_matches']:
        raise CensusError('class-segmentation archive MD5 does not match NAVER published digest')
    palette = _parse_palette(textgt_tar)
    expected_frames = _metadata_frame_keys(textgt_tar)
    rgb_sel, seg_sel = {}, {}
    for key, member in rgb_keys.items():
        scene, cond, cam, _ = key
        if scene in ROLE and cond in CONDITIONS and cam == 0:
            rgb_sel[key] = member
    for key, member in seg_keys.items():
        scene, cond, cam, _ = key
        if scene in ROLE and cond in CONDITIONS and cam == 0:
            seg_sel[key] = member
    keyset = set(rgb_sel) | set(seg_sel)
    per_role = {r: {c: {'rgb_members': 0, 'segmentation_members': 0, 'matched': 0,
                        'rgb_without_mask': 0, 'mask_without_rgb': 0, 'scene_sequences': 0,
                        'sequences_eligible_30': 0, 'native_index_gap_count': 0,
                        'sequences_with_native_gaps': 0, 'paired_with_clone_fog': 0,
                        'paired_with_clone_rain': 0}
                  for c in CONDITIONS} for r in sorted(set(ROLE.values()))}
    per_scene = []
    selected_groups = {}
    for scene in SCENES:
        for cond in CONDITIONS:
            all_seq = [k for k in keyset if k[0] == scene and k[1] == cond]
            matches = [k for k in all_seq if k in rgb_sel and k in seg_sel]
            rgb_only = sum(k in rgb_sel and k not in seg_sel for k in all_seq)
            mask_only = sum(k in seg_sel and k not in rgb_sel for k in all_seq)
            key = (scene, cond)
            selected_groups[key] = set(k[3] for k in matches)
            expected = expected_frames[(scene, cond)]
            observed = {k[3] for k in matches}
            missing_expected = expected - observed
            unexpected_observed = observed - expected
            if unexpected_observed:
                raise CensusError('image archives contain frame keys absent from publisher camera metadata')
            seq = _sequence_summary(matches, expected)
            role = ROLE[scene]
            agg = per_role[role][cond]
            agg['rgb_members'] += sum(k[0] == scene and k[1] == cond for k in rgb_sel)
            agg['segmentation_members'] += sum(k[0] == scene and k[1] == cond for k in seg_sel)
            agg['matched'] += len(matches)
            agg['rgb_without_mask'] += rgb_only
            agg['mask_without_rgb'] += mask_only
            agg['scene_sequences'] += 1
            agg['sequences_eligible_30'] += int(seq['eligible_30'])
            agg['native_index_gap_count'] += seq['native_index_gap_count'] or 0
            agg['sequences_with_native_gaps'] += int(not seq['contiguous_from_zero'] and bool(matches))
            per_scene.append({'scene': scene, 'role': role, 'condition': cond, **seq,
                              'expected_camera0_frames': len(expected),
                              'missing_pairs_from_metadata': len(missing_expected),
                              'rgb_without_mask': rgb_only, 'mask_without_rgb': mask_only})
    for scene in SCENES:
        clone = selected_groups[(scene, 'clone')]
        for cond in ('fog', 'rain'):
            paired = len(clone & selected_groups[(scene, cond)])
            per_role[ROLE[scene]][cond]['paired_with_clone_fog' if cond == 'fog' else 'paired_with_clone_rain'] += paired
    sample_candidates = [k for k in sorted(seg_sel) if k[0:2] == PALETTE_SCENE_CONDITION and k[2] == 0]
    if not sample_candidates:
        raise CensusError('no development-only sample mask is available for the palette check')
    sample_mask_name = seg_sel[sample_candidates[0]]
    palette_report = _palette_check(seg_path, sample_mask_name, palette)
    if not palette_report['matches_publisher_palette']:
        raise CensusError(f"development mask contains {palette_report['unexpected_rgb_color_count']} colors outside publisher palette")
    # One-based lower bound of 30 adjacent labelled steps; windows and weather copies never create new independent scenes.
    summary = {
        'dataset': 'Virtual KITTI 2 v2.0.3', 'license': 'CC BY-NC-SA 3.0; non-commercial use',
        'publisher': 'NAVER LABS Europe / Naver Corporation',
        'archive_digests': {'rgb': rgb_meta, 'class_segmentation': seg_meta},
        'archive_member_counts': {'rgb': rgb_members, 'class_segmentation': seg_members},
        'publisher_metadata_sha256': file_hashes(textgt_tar)['sha256'],
        'protocol': {'camera': 0, 'conditions': list(CONDITIONS), 'scene_roles': ROLE,
                     'scene_id_member_names_exposed': False, 'frame_member_names_exposed': False,
                     'held_out_mask_pixels_decoded': False},
        'scope_counts': per_role, 'per_scene_condition_counts': per_scene,
        'total_selected_rgb': len(rgb_sel), 'total_selected_segmentation': len(seg_sel),
        'total_matched': len(set(rgb_sel) & set(seg_sel)),
        'total_rgb_without_mask': len(set(rgb_sel) - set(seg_sel)),
        'total_mask_without_rgb': len(set(seg_sel) - set(rgb_sel)),
        'palette_check': palette_report,
        'independent_evaluation_scenes': 2,
        'inference_scope': 'descriptive simulated weather-shift replay only; no claim of real-weather generalization',
    }
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb', required=True)
    parser.add_argument('--segmentation', required=True)
    parser.add_argument('--textgt', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--allow-nonpublisher-md5', action='store_true', help='synthetic tests only')
    args = parser.parse_args()
    result = census(args.rgb, args.segmentation, args.textgt, strict_md5=not args.allow_nonpublisher_md5)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'status': 'ok', 'total_matched': result['total_matched'],
                      'unexpected_palette_colors': result['palette_check']['unexpected_rgb_color_count'],
                      'report': str(args.out)}, indent=2))

if __name__ == '__main__':
    try:
        main()
    except CensusError as exc:
        raise SystemExit(f'VKITTI census failed: {exc}')

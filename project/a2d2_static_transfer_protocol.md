# A2D2 · fixed static transfer check

The metadata handback is accepted. The saved census SHA-256 matches `7543b5a4b916b6b9a8a91d4e29828e095048184bd328432a5e0af44b7049da7e`. The 2,796 pairs outside exact triplets are same-ID image/mask pairs from two folders missing camera JSON. This phase excludes them under the exact-triplet rule. The 63 timestamp spot checks do not establish a labelled time series. This is a static real-image transfer check from a simulator-trained model. It does not measure a weather effect, feedback, causal adaptation, or independent-route performance. The open-data registry and publisher license permit public access; retain pixels and transformed imagery locally and cite the dataset. [AWS Open Data registry](https://registry.opendata.aws/aev-a2d2/) · [Publisher semantic documentation](https://audi-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/camera_lidar_semantic/README-SemSeg.txt)

## Fixed sample and mapping

Relist exact front-center image, semantic mask and camera-JSON triplets on the official extracted mirror. Require the 21 nonempty folder inventories and 28,652 total exact triplets to match the retained census. Select four IDs per nonempty folder, ordered by SHA-256 of UTF-8 `a2d2-static-v1` + NUL + full publisher scene key + NUL + zero-padded nine-digit ID, with numeric ID as tie-breaker. Select the lowest four digests per folder. Lock and hash the resulting 84-row manifest before requesting pixels. Record publisher object lengths and ETags and verify the object snapshots at download, then hash the received bytes. Missing, corrupt or changed objects block the run, without replacement. Never fetch the full archive for this check.

Map these exact publisher class names to the existing VKITTI output names and indices. All recognized source categories outside the table are ignored for the mapped score. Validate RGB masks against the exact publisher class-list bytes bound to the census. Fail on an unknown color or dimension mismatch. Do not infer class equivalence from a visual example.

| A2D2 | VKITTI class (model index) |
| --- | --- |
| Buildings | Building (4) |
| RD normal street | Road (5) |
| Traffic sign 1, 2, 3 | TrafficSign (7) |
| Traffic signal 1, 2, 3 | TrafficLight (8) |
| Poles | Pole (9) |
| Truck 1, 2, 3 | Truck (11) |
| Car 1, 2, 3, 4 | Car (12) |
| Sky | Sky (1) |

Leave road variants, other vehicles, vegetation and all other ambiguous categories outside the map. Keep all 14 logits as competitors and preserve their probability mass. Top-1 in an unmapped class is an error on a mapped ground-truth pixel. Report the mapped-pixel fraction and support per class.

## Scoring and handback

Use the fixed epoch-6 VKITTI checkpoint and exact geometry restoration and scoring code, with the verified Scene18 threshold `0.9931015335023403`. No fit, target-domain calibration, thresholds selected from A2D2, or PID/ACI replay. For mapped pixels compute top-1 error, 5% miss-frame breach at the fixed threshold (strict score > threshold), and mean set size. Report sample frame count, mapped and native pixel counts, per-class support and top-1 error counts, per-folder and per-date breach numerators/denominators, and an equal-date descriptive mean across dates with scorable images. Keep dates separate and report them as coarse reporting strata, not independent routes. A sample with zero mapped pixels is explicitly unscorable; do not replace it or silently count it as a success. No significance test or population coverage claim.

Build a small fixture for palette decoding, different native dimensions, excluded classes, all-ignored masks, reproducible sample selection and no target calibration. Record source, manifest, checkpoint, calibration, driver and result hashes with runtime identity. Run once when those checks pass. Keep raw images, masks, per-frame records and any derived visual material local. Return aggregate transfer results and clear constraints for journal use.

# Reproducing the experiments

The synthetic example in the README requires no external data. This page describes the original study drivers. They bind an exact trained model and calibration vector before evaluating. The original model and run records are retained by the authors. They are not in this repository. Re-fitting from the same seed on other software or hardware can yield different bytes. In that case the frozen evaluation drivers stop at their provenance checks. A new fit would be a replication with a new model, not a rerun of the original checkpoint.

## Runtime

The recorded study environment was Python 3.12.8 on arm64 macOS, CPU, NumPy 2.5.0, Pillow 12.2.0, PyTorch 2.12.1 and torchvision 0.27.1. Exact versions are listed in `requirements-study.txt`. The original package archive hashes were unavailable. Use a separate virtual environment. Check the vendor's installation instructions for your platform if its indexes do not carry these versions.

## Virtual KITTI 2: inventory and fit

Obtain the RGB, class segmentation and text archives from the [NAVER release](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/), under CC BY-NC-SA 3.0. Leave the archives intact. Put them in `.local-data/vkitti2/` at the root of this checkout. From the checkout root, using Bash or Zsh:

```sh
vk=.local-data/vkitti2
vk_args=(--rgb "$vk/vkitti_2.0.3_rgb.tar"
         --segmentation "$vk/vkitti_2.0.3_classSegmentation.tar"
         --textgt "$vk/vkitti_2.0.3_textgt.tar.gz")
python stage0/vkitti_census.py "${vk_args[@]}" --out "$vk/census.json"
python stage0/vkitti_train.py preflight "${vk_args[@]}" --out "$vk/training_preflight.json"
python stage0/vkitti_train.py train "${vk_args[@]}" --out "$vk/training"
```

See the [training recipe](../project/vkitti_training_recipe.md) for train-only scenes and settings. The [calibration and replay protocol](../project/vkitti_calibration_replay_protocol.md) specifies the native-resolution scoring rule, seven controllers, held-out scenes and metrics.

## Replay the frozen checkpoint

Use the verified original checkpoint and receipt, or locally recreated artifacts that pass the frozen checks. The held-out driver expects the weights, fit receipt, census, calibration arrays, original calibration receipt, and a scope correction in the locations listed in `stage0/vkitti_heldout.py`. When using a new local calibration receipt from the following exact calibration command, use the receipt preparation helper to preserve its bytes and record the distinction between calibration and held-out access.

```sh
python stage0/vkitti_score.py calibrate "${vk_args[@]}" \
  --weights "$vk/training/vkitti2_deeplab_mobilenetv3_seed101_epoch6.npz" \
  --training-record "$vk/training/training_record.json" --out "$vk/calibration"
python stage0/prepare_calibration_receipt.py "$vk/calibration"
python stage0/vkitti_heldout.py preflight
python stage0/vkitti_heldout.py run
```

On checkouts with the authors' archived receipts from the original study, retain the original hashed protocol bytes alongside those receipts. The protocol files published here omit local administrative instructions. This release omits two internal metadata flags from the replay and transfer drivers. Those omissions affect file and record hashes only. For an authorized rerun with the public source, generate new preflight and run receipts under that source. The driver records its source digests and uses a single-use run directory.

## Static A2D2 transfer

The [A2D2 registry](https://registry.opendata.aws/aev-a2d2/) documents the public mirror and CC BY-ND 4.0 license. See the [sampling and mapping protocol](../project/a2d2_static_transfer_protocol.md). The code requires the original fixed checkpoint and Scene18 anchor, reproduces the exact-triplet selection from the mirror, and freezes object identities before retrieval. Reproduction is conditional on the mirror inventory and objects remaining available.

```sh
a2=.local-data/a2d2
mkdir -p "$a2/static_transfer"
curl -fL -o "$a2/static_transfer/class_list.json" \
  https://audi-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/camera_lidar_semantic/class_list.json
python stage0/a2d2_metadata_census.py --out-dir "$a2/metadata"
python stage0/a2d2_static_transfer.py self-test --class-list "$a2/static_transfer/class_list.json" \
  --out "$a2/static_transfer/self_test.json"
python stage0/a2d2_static_transfer.py lock-sample --class-list "$a2/static_transfer/class_list.json" \
  --out "$a2/static_transfer/sample_manifest.private.json"
python stage0/a2d2_static_transfer.py run --class-list "$a2/static_transfer/class_list.json" \
  --manifest "$a2/static_transfer/sample_manifest.private.json" --out-dir "$a2/static_transfer/run"
```

## Results available in this checkout

[Aggregate results](../results/descriptive_results.json) contains all 42 Virtual KITTI 2 scene/condition/method rows and the A2D2 sample and class totals. It contains no image or mask pixels, A2D2 frame selections, per-frame scores, thresholds, or camera metadata. These are descriptive summaries. Set size includes every native pixel in the replay and only mapped pixels for A2D2. Neither experiment estimates coverage for the real-driving population.

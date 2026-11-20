# Computer Vision Group Project

Research code for an exploratory study of online conformal prediction sets for semantic segmentation. The study used controlled fog and rain render shifts in Virtual KITTI 2 and a separate fixed-threshold transfer check on a sample of A2D2 images. The two experiments measure different questions.

This repository contains the scoring and controller primitives, a synthetic replay example, and the Virtual KITTI 2 archive metadata census. It is a code snapshot. The full experiment artifacts and manuscript are maintained separately.

## Run the synthetic replay

Use Python 3.12 and install NumPy. The archive census also uses Pillow to check image and mask encodings.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install numpy Pillow
python replay/run.py examples/synthetic_replay.json /tmp/confseg-decisions.csv
```

The example uses generated scores and features. It needs no datasets or trained weights. It writes the threshold trajectory to CSV and evaluator metrics to a separate JSON file.

- `core/confseg_core/` implements scores, prediction-set statistics and delayed-feedback controllers.
- `replay/` includes the synthetic CLI and the clear-prefix replay protocol.
- `stage0/vkitti_census.py` checks archive hashes, frame pairing and palette metadata against the original publisher archives.

## Datasets and references

- [Virtual KITTI 2](https://europe.naverlabs.com/proxy-virtual-worlds-vkitti-2/) · [Cabon et al.](https://arxiv.org/abs/2001.10773). NAVER LABS Europe publishes the dataset under CC BY-NC-SA 3.0.
- [A2D2](https://registry.opendata.aws/aev-a2d2/) · [Geyer et al.](https://arxiv.org/abs/2004.06320). The dataset is published under CC BY-ND 4.0.
- [Adaptive Conformal Inference Under Distribution Shift](https://proceedings.neurips.cc/paper_files/paper/2021/file/0d441de75945e5acbc865406fc9a2559-Paper.pdf) · [Conformal PID Control for Time Series Prediction](https://proceedings.neurips.cc/paper_files/paper/2023/file/47f2fad8c1111d07f83c91be7870f8db-Paper-Conference.pdf).

# snn-csi-tracking

Spiking neural networks for human presence detection and position tracking from Wi-Fi Channel State Information (CSI).

The task: given a stream of raw CSI (per-antenna, per-subcarrier channel measurements from a Wi-Fi link), predict whether a person is present in the room and, if so, roughly where — without a camera or dedicated positioning sensor at inference time. The model is a small spiking network (leaky-integrate-and-fire units) fed by a threshold-based spike encoding of how the channel changes frame to frame, motivated by the sparsity and energy efficiency that gives over a conventional dense network of the same size.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python ≥ 3.10. Core dependencies: PyTorch, [snnTorch](https://snntorch.readthedocs.io/), NumPy/SciPy/pandas/scikit-learn, MediaPipe (ground-truth trajectory extraction), OpenCV, and `transformers` (metric depth estimation for the 3D variant).

## Data

Not included in this repository. The pipeline expects raw CSI captures and matching videos under `data/raw_captures/`, one folder per `{condition}_{activity}` combination:

```
data/raw_captures/
    {condition}_{activity}/
        capture1.mat ... capture5.mat   # plain-text CSI, one line per (frame, antenna)
        videos/
            segment1.mp4 ... segment5.mp4
```

`condition` is `NLoS` or `PLoS` (line-of-sight blocked or not); `activity` is one of `E` (empty room), `EN-S`, `EN-W`, `L`, `S`. See `src/snn_csi_tracking/data/raw_capture_loader.py` for the exact file format. Extracted features, trajectories, and depth maps are cached under `data/processed/` on first run.

## Running the pipeline

Train the main model (2D position, matched SNN/ANN pair for the energy comparison):

```bash
.venv/bin/python src/snn_csi_tracking/training/train_presence_position_conv.py --model snn
.venv/bin/python src/snn_csi_tracking/training/train_presence_position_conv.py --model ann
```

`--use-3d` adds real camera-relative depth (pinhole projection); `--use-3d --no-pinhole` keeps normalized image-plane `(x, y)` and appends a normalized depth channel.

Leave-one-activity-out cross-validation (the honest generalization number, given only 10 recording sessions total):

```bash
.venv/bin/python scripts/train_presence_position_cv.py --model snn
.venv/bin/python scripts/train_presence_position_cv.py --model ann
```

Energy estimate (spiking vs. dense, per Horowitz-style per-operation energy costs):

```bash
.venv/bin/python scripts/estimate_energy_presence_position.py
.venv/bin/python scripts/estimate_energy_lstm.py
```

Other scripts under `scripts/` cover evaluation of a saved checkpoint (`evaluate_presence_position.py`), plotting (`plot_rmse_vs_epoch*.py`, `plot_energy_comparison.py`, `plot_features_enw_paper.py`), and video overlays / trajectory visualization for qualitative inspection (`render_*.py`, `visualize_trajectory_prediction.py`).

## Layout

```
src/snn_csi_tracking/
    data/          # raw CSI parsing, feature extraction, windowing, trajectory/depth extraction
    models/         # SNN / ANN / LSTM architectures, spike encoding
    training/       # training entry points
    inference.py    # post-processing (presence debouncing, position smoothing)
scripts/            # training/CV runs, energy estimation, plotting, evaluation, visualization
```

## License

MIT — see [LICENSE](LICENSE).

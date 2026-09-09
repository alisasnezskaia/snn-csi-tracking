# Experiment log — motion/speed regression (`train_regression.py`)

Tracks what's been tried on the CSI -> speed regression task, in order, with
results and verdicts. Update this whenever a config is trained to completion
-- the goal is to never re-run (or re-argue) something we already measured.

All runs use the same 171-trial dataset (`data/processed/regression_rates...`
cache) and the same train/val/test split (chronological, per-trial, 70/15/15)
unless noted otherwise. Metric: test MSE, and R^2 = 1 - test_mse/label_variance
(label variance ~17.3-17.4 depending on cache).

## Current best-known config (what's in the repo right now)

- `STRIDE = 64`, rate conditioning on (embed `rate_idx`, concat to every
  timestep's spikes), no dropout, no weight decay, 20 epochs.
- `PerFrameConvEncoder`: flatten (not average-pool) -> 8192-dim per-frame
  feature.
- Readout: last-timestep membrane potential only (not averaged over time).
- **test_mse = 8.7375, R^2 = 0.499**

## Runs, in order

### 1. Baseline: stride=32, no rate conditioning, 20 epochs
- Every capture rate (5/10/50/100ms) pooled together with no way for the
  model to tell them apart; window stride 32 (75% overlap, ~22k windows from
  171 trials).
- **test_mse = 11.3889, R^2 = 0.342**
- Diagnosis: heavy window overlap on a small number of independent trials
  (171) let the model partially memorize per-trial CSI fingerprints rather
  than learn the general mapping; pooling 4 different real-world-duration
  capture rates as if equivalent (a 128-sample window spans 0.64s at 5ms vs
  12.8s at 100ms) actively hurt rather than helped.

### 2. Stride 64 + rate conditioning (BEST SO FAR)
- Changed `STRIDE` 32 -> 64 (halves window overlap/redundancy).
- Added a learned embedding for the discrete capture-rate index, concatenated
  to every timestep's spike vector before the hidden LIF stack.
- **test_mse = 8.7375, R^2 = 0.499** (best epoch 10/20, val_mse=13.4631)
- Verdict: **both changes together gave a large, real improvement**
  (R^2 0.342 -> 0.499). Not decomposed into "which change helped how much"
  individually -- if isolating matters later, that's a cheap follow-up
  (run stride=64 alone, then rate-conditioning alone).

### 3. + dropout (0.2) + weight decay (1e-4), 40 epochs
- Motivation: run #2 still peaked at epoch 10/20 then overfit for the rest,
  so tried regularizing to let training keep improving longer.
- **test_mse = 8.8428, R^2 = 0.493** (best epoch 21/40, val_mse=13.9219)
- Verdict: **no real improvement** (within noise of run #2, arguably
  slightly worse). Best epoch did move later (10 -> 21), confirming the
  regularization delayed overfitting as intended, but the val/test floor
  didn't drop. Conclusion: regularization strength wasn't the bottleneck --
  pointed at a structural capacity issue instead (see #4's motivation).
- **Reverted** -- current code has `DROPOUT=0.0`, `WEIGHT_DECAY=0.0`,
  `NUM_EPOCHS=20`.

### 4. + global-average-pool conv encoder + readout averaged over all timesteps
- Motivation: `PerFrameConvEncoder`'s flatten produces an 8192-dim per-frame
  feature (32 channels x 256 pooled subcarriers) feeding a >1M-parameter
  first hidden layer -- likely enough spare capacity to memorize per-trial
  noise regardless of dropout/weight decay. Replaced flatten with
  `x.mean(dim=-1)` (global average pool over subcarriers, 8192 -> 32 dims).
  Also changed the readout from "last timestep's membrane potential only" to
  "average membrane potential across all 128 timesteps", since the original
  only used the final, heavily beta-decayed timestep.
- **test_mse = 11.2435, R^2 = 0.356** -- clearly worse, not just flat.
- Verdict: **regression, not an improvement.** Reverted both changes.
  Not decomposed into which of the two hurt more -- worth isolating before
  trying either idea again (test avg-pool alone, and readout-averaging
  alone, rather than both at once). Hypothesis for why avg-pool may have
  hurt: collapsing all 256 pooled-subcarrier positions to a single mean per
  channel may throw away real spatial (subcarrier-position) information the
  8192-dim version was actually using, not just redundant noise as assumed.
- **Reverted** -- current code has the flatten-based encoder and
  last-timestep-only readout back.

## Ideas discussed but not yet tried (backlog)

- **Isolate #4's two changes** (avg-pool alone vs. readout-averaging alone)
  before concluding either is a dead end outright.
- **CSI phase / Doppler features** instead of amplitude-only (`normalize()`
  currently does `np.abs()`, discarding phase entirely). Highest-ceiling
  option discussed -- phase/Doppler shift is physically ~proportional to
  velocity, unlike amplitude. Requires phase sanitization (CFO/SFO removal
  or cross-antenna phase difference) before it's usable.
- **Domain-adversarial regularization against trial identity** -- add an
  auxiliary head predicting `trial_id` from the shared representation with a
  gradient-reversal layer, to directly penalize the memorization shortcut
  diagnosed in run #1/#3, rather than generic weight regularization.
- **Held-out-trial split** instead of the current chronological-per-trial
  split -- current split lets every trial appear in train/val/test, which
  likely lets partial trial-identity memorization "pay off" on val/test too;
  a true held-out-trial split would be a harder but more honest number.
- **Temporal attention / transformer readout** over the 128 timesteps,
  instead of (or alongside) the sequential LIF recurrence -- see
  "WiTransformer" precedent for WiFi CSI.
- **Ground-truth extraction quality**: switched video trajectory extraction
  from background-subtraction to MediaPipe pose (hip -> nose fallback) with
  ROI exclusion (near-camera equipment desk) and Kalman-filter gap-filling
  (`build_trajectory_labels.py`) -- implemented, not yet run as a full batch
  or validated against the old background-subtraction labels.
- **Real-world (meter) calibration** of the position/speed labels via a
  floor-plane homography (`calibrate_homography.py` +
  `apply_homography_to_trajectories.py`, built, not yet run) -- would
  replace the current uncalibrated pixels/frame label with actual m/s.
- **Recover the 17 trials missing ground truth** (13 of them NLoS/5ms) --
  modest size gain (~11%) but fixes a real coverage gap in that specific
  condition/rate combination.

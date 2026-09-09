





"""Torch Dataset wrapping windowed CSI samples.

Decoupled from the .mat schema: build_datasets() takes an already-prepared
list of (window, label, group) triples (e.g. produced by
mat_loader.load_record() + preprocessing.sliding_windows()), so this module
doesn't need to change when the field names in mat_loader are filled in.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class CSIDataset(Dataset):
    """Keeps `windows` as-is (may be a disk-backed memmap covering the full
    dataset) and only materializes one sample at a time in __getitem__, via
    `indices` into it. Fancy-indexing a memmap with the whole split up front
    (windows[indices]) would force the entire split into RAM at once --
    with train+val+test together that's ~the whole dataset, which is what
    caused the OOM kills on this machine."""

    def __init__(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        label_dtype: torch.dtype = torch.long,
        extra: np.ndarray | None = None,
        extra2: np.ndarray | None = None,
    ):
        """extra: optional per-window side info (e.g. a rate-index array),
        same length as `labels`. When given, __getitem__ yields a 3-tuple
        (window, label, extra) instead of the usual 2-tuple -- every caller
        of a given CSIDataset instance sees a consistent arity.

        extra2: optional SECOND per-window side info (e.g. a condition/LoS
        label), only usable together with `extra` -- when given, __getitem__
        yields a 4-tuple (window, label, extra, extra2). Kept as a separate
        slot rather than packed into `extra` so existing 3-tuple callers
        (extra2=None, the default) are completely unaffected."""
        if len(windows) != len(labels):
            raise ValueError(f"Got {len(windows)} windows but {len(labels)} labels")
        if extra2 is not None and extra is None:
            raise ValueError("extra2 requires extra to also be given")
        self.windows = windows
        self.indices = indices
        self.labels = torch.as_tensor(labels[indices], dtype=label_dtype)
        self.extra = torch.as_tensor(extra[indices]) if extra is not None else None
        self.extra2 = torch.as_tensor(extra2[indices]) if extra2 is not None else None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        window = np.asarray(self.windows[self.indices[idx]])
        window_t = torch.as_tensor(window, dtype=torch.float32)
        if self.extra2 is not None:
            return window_t, self.labels[idx], self.extra[idx], self.extra2[idx]
        if self.extra is not None:
            return window_t, self.labels[idx], self.extra[idx]
        return window_t, self.labels[idx]


def build_datasets(
    windows: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    val_size: float = 0.15,
    test_size: float = 0.15,
    label_dtype: torch.dtype = torch.long,
    extra: np.ndarray | None = None,
) -> tuple[CSIDataset, CSIDataset, CSIDataset]:
    """Chronological split of pre-windowed samples into train/val/test CSIDatasets.

    `groups` gives one id per source trial, and windows within a trial are in
    time order (see build_windows_and_labels/preprocessing.sliding_windows).
    For each trial independently: samples in [0, t1) -> train, [t1, t2) ->
    val, [t2, T) -> test, where t1/t2 are that trial's own length scaled by
    (1 - val_size - test_size) and (1 - test_size). This keeps every trial
    represented in all three splits while respecting time order (no
    training on the future to predict the past), unlike a random shuffle.
    """
    train_idx, val_idx, test_idx = [], [], []
    for group_id in np.unique(groups):
        (group_positions,) = np.where(groups == group_id)
        n = len(group_positions)
        t1 = int(round(n * (1 - val_size - test_size)))
        t2 = int(round(n * (1 - test_size)))
        train_idx.append(group_positions[:t1])
        val_idx.append(group_positions[t1:t2])
        test_idx.append(group_positions[t2:])

    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    test_idx = np.concatenate(test_idx)

    return (
        CSIDataset(windows, labels, train_idx, label_dtype, extra=extra),
        CSIDataset(windows, labels, val_idx, label_dtype, extra=extra),
        CSIDataset(windows, labels, test_idx, label_dtype, extra=extra),
    )


def build_datasets_grouped(
    windows: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    val_size: float = 0.15,
    test_size: float = 0.15,
    label_dtype: torch.dtype = torch.long,
    extra: np.ndarray | None = None,
    strata: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[CSIDataset, CSIDataset, CSIDataset]:
    """Held-out-trial split: each whole trial (one `groups` id) goes entirely
    into train, val, or test -- unlike build_datasets' within-trial
    chronological split, no trial ever contributes windows to more than one
    set. This is the honest-evaluation split: a model can't score well by
    keying off one trial's session-specific hardware-drift/background
    fingerprint, since that fingerprint never appears on both sides of the
    split (unlike a chronological split, which leaks it).

    `strata`: optional per-window array (same length as `groups`, constant
    within a trial -- e.g. activity code) used to split each stratum's
    trials independently and recombine, so every stratum is represented
    across train/val/test rather than risking a rare one (e.g. an activity
    with only 1-2 trials) landing entirely in test by chance. Strata with
    too few trials to split cleanly are handled gracefully (n=1 -> train
    only; n=2 -> train+test, no val) rather than raising -- with a dataset
    this small, some strata just can't be spread across all three sets.
    Pass None to split all trials as one pool with no stratification.
    """
    rng = np.random.RandomState(seed)
    unique_groups = np.unique(groups)

    if strata is not None:
        first_seen: dict = {}
        for i, g in enumerate(groups):
            first_seen.setdefault(g, strata[i])
        buckets: dict = {}
        for g in unique_groups:
            buckets.setdefault(first_seen[g], []).append(g)
    else:
        buckets = {None: list(unique_groups)}

    train_groups, val_groups, test_groups = [], [], []
    for stratum_groups in buckets.values():
        stratum_groups = np.array(stratum_groups)
        rng.shuffle(stratum_groups)
        n = len(stratum_groups)
        n_test = max(1, round(n * test_size)) if n >= 2 else 0
        n_val = max(1, round(n * val_size)) if n - n_test >= 2 else 0
        test_groups.extend(stratum_groups[:n_test])
        val_groups.extend(stratum_groups[n_test : n_test + n_val])
        train_groups.extend(stratum_groups[n_test + n_val :])

    train_idx = np.where(np.isin(groups, train_groups))[0]
    val_idx = np.where(np.isin(groups, val_groups))[0]
    test_idx = np.where(np.isin(groups, test_groups))[0]

    return (
        CSIDataset(windows, labels, train_idx, label_dtype, extra=extra),
        CSIDataset(windows, labels, val_idx, label_dtype, extra=extra),
        CSIDataset(windows, labels, test_idx, label_dtype, extra=extra),
    )

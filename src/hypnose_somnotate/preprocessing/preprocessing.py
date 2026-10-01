"""Preprocessing helpers built on the Somnotate pipeline."""

from __future__ import annotations

import numpy as np

from ..somnotate_pipeline.preprocessing.preprocess_signals import compute_log_spectrogram, preprocess
from ..somnotate_pipeline.utils.configuration import time_resolution
from ..somnotate._utils import _truncate_signals
from .gap_correction import PreparedRecording, epoch_kinds

LOW_CUT = 1.0
HIGH_CUT = 90.0
NOTCH_LOW_CUT = 45.0
NOTCH_HIGH_CUT = 55.0

NormalizationStats = tuple[np.ndarray, np.ndarray]


def preprocess_multichannel(
    raw_signals: np.ndarray,
    sampling_rate_hz: float,
    *,
    normalization_stats: list[NormalizationStats | None] | None = None,
) -> np.ndarray:
    """Preprocess every channel of `raw_signals` into its normalized spectrogram.

    normalization_stats
        Optional, one entry per channel (matching `raw_signals`' columns).
        Each entry is either ``None`` (fall back to this call's own
        percentile-trimmed mean/std, the original per-chunk behaviour) or an
        ``(robust_mean, robust_std)`` pair -- typically from
        `compute_global_normalization_stats` -- applied instead. Passing
        `None` for the whole argument is the same as an all-`None` list.
    """
    preprocessed_signals = []
    for ch, signal in enumerate(raw_signals.T):
        stats = normalization_stats[ch] if normalization_stats is not None else None
        robust_mean, robust_std = stats if stats is not None else (None, None)
        _, _, preprocessed_signal = preprocess(
            signal,
            sampling_rate_hz,
            time_resolution_in_sec=time_resolution,
            low_cut=LOW_CUT,
            high_cut=HIGH_CUT,
            notch_low_cut=NOTCH_LOW_CUT,
            notch_high_cut=NOTCH_HIGH_CUT,
            robust_mean=robust_mean,
            robust_std=robust_std,
        )
        preprocessed_signals.append(preprocessed_signal)

    return np.concatenate([signal.T for signal in preprocessed_signals], axis=1)


def compute_global_normalization_stats(
    prepared: PreparedRecording,
    sampling_rate_hz: float,
    *,
    p: float = 5.0,
) -> list[NormalizationStats | None]:
    """Pool robust normalization statistics across every scoring chunk of a recording.

    `preprocess_multichannel`'s default behaviour normalizes each
    `ScoringChunk` against its own percentile-trimmed mean/std. That is
    fine for a single-chunk recording (`trim_only`/`mask_inline`), but under
    `split` -- several chunks cut apart by long gaps -- it means the same
    underlying physiology is scored against a different baseline depending
    on which chunk it fell into, and even a single `mask_inline` chunk's
    statistics still shift with how long or how gap-heavy that recording is.

    This computes one set of statistics per channel from every "signal"
    epoch across *all* of `prepared.scoring_chunks` pooled together --
    "gap"/"too_short" epochs (see `epoch_kinds`) never contribute, so
    zero-padded gaps can't bias the estimate. Pass the result to
    `preprocess_multichannel(..., normalization_stats=...)` so every chunk
    is normalized against the same, recording-wide baseline regardless of
    how many pieces gaps split it into.

    Returns
    -------
    list of length `n_channels`; each entry is `(robust_mean, robust_std)`
    (one value per retained frequency bin) or `None` if that channel had no
    "signal" epochs anywhere in the recording -- `preprocess_multichannel`
    falls back to local per-chunk normalization for that channel.
    """
    if not prepared.scoring_chunks:
        return []

    kinds = epoch_kinds(prepared)
    time_res = prepared.time_resolution_s
    n_channels = prepared.scoring_chunks[0].raw_signal.shape[1]
    pooled_per_channel: list[list[np.ndarray]] = [[] for _ in range(n_channels)]

    for chunk in prepared.scoring_chunks:
        chunk_start_ep = int(round(chunk.original_start_s / time_res))
        for ch in range(n_channels):
            _, _, log_spectrogram = compute_log_spectrogram(
                chunk.raw_signal[:, ch],
                sampling_rate_hz,
                time_resolution_in_sec=time_res,
                low_cut=LOW_CUT,
                high_cut=HIGH_CUT,
                notch_low_cut=NOTCH_LOW_CUT,
                notch_high_cut=NOTCH_HIGH_CUT,
            )
            n_ep = log_spectrogram.shape[1]
            chunk_kinds = kinds[chunk_start_ep:chunk_start_ep + n_ep]
            signal_mask = chunk_kinds == "signal"
            if np.any(signal_mask):
                pooled_per_channel[ch].append(log_spectrogram[:, signal_mask])

    stats: list[NormalizationStats | None] = []
    for ch in range(n_channels):
        if not pooled_per_channel[ch]:
            stats.append(None)
            continue
        pooled = np.concatenate(pooled_per_channel[ch], axis=1)
        robust_mean = np.empty(pooled.shape[0])
        robust_std = np.empty(pooled.shape[0])
        for f in range(pooled.shape[0]):
            truncated = _truncate_signals(pooled[f].copy(), p, 100.0 - p)
            robust_mean[f] = np.mean(truncated)
            robust_std[f] = np.std(truncated)
        stats.append((robust_mean, robust_std))
    return stats

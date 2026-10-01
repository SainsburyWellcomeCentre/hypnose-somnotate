"""Preprocessing helpers built on the Somnotate pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np

from ..somnotate_pipeline.preprocessing.preprocess_signals import compute_log_spectrogram, preprocess
from ..somnotate_pipeline.utils.configuration import time_resolution
from ..somnotate._utils import _truncate_signals
from .gap_correction import PreparedRecording, epoch_kinds

LOW_CUT = 1.0
HIGH_CUT = 90.0
NOTCH_LOW_CUT = 45.0
NOTCH_HIGH_CUT = 55.0
# Percentile trim used for every pooled robust mean/std.
ROBUST_TRIM_PERCENT = 5.0

NormalizationStats = tuple[np.ndarray, np.ndarray]
ChannelStats = list[Union[NormalizationStats, None]]
# A fixed set of per-channel stats, or a callable that picks them once the
# recording's gap/chunk plan is known (returning None to fall back).
StatsSource = Union[ChannelStats, Callable[[PreparedRecording], Union[ChannelStats, None]]]

NORMALIZATION_STATS_FORMAT = "hypnose-somnotate-normalization/1"


@dataclass
class NormalizationResult:
    """Which statistics a scored recording was normalized against.

    source
        ``"reference"`` -- statistics supplied by the caller (typically pooled
        from another, longer recording of the same animal); ``"self"`` -- the
        recording's own pooled statistics (``global_normalization``);
        ``"local"`` -- each scoring chunk against its own statistics.
    applied
        The per-channel stats actually used, or None for ``"local"``.
    own
        The recording's own pooled statistics, always computed, so a caller
        can save them as a future reference or compare them with ``applied``.
    signal_s
        Seconds of scoreable ("signal") epochs the own statistics pool.
    """

    source: str
    applied: ChannelStats | None
    own: ChannelStats
    signal_s: float

    def to_dict(self) -> dict:
        """JSON-serializable summary (no statistics arrays)."""
        return {"source": self.source, "signal_s": float(self.signal_s)}


def signal_duration_s(prepared: PreparedRecording) -> float:
    """Seconds of the recording scored as signal (gaps, artifacts, too_short excluded)."""
    return float(sum(s.duration_s for s in prepared.segments if s.kind == "signal"))


def normalization_offset_z(own: ChannelStats, reference: ChannelStats) -> list[float | None]:
    """Per channel, how far a recording sits from a reference baseline, in reference SDs.

    The median over frequency bins of ``(own_mean - ref_mean) / ref_std``.
    Differences in sleep-state mix move individual bands in opposite
    directions and largely cancel in the median; a gain or impedance change
    shifts every bin of the log-spectrum the same way and does not -- which is
    the case where borrowing a reference baseline is unsafe. None for a
    channel missing from either side.
    """
    offsets: list[float | None] = []
    for ch in range(max(len(own), len(reference))):
        own_ch = own[ch] if ch < len(own) else None
        ref_ch = reference[ch] if ch < len(reference) else None
        if own_ch is None or ref_ch is None:
            offsets.append(None)
            continue
        ref_std = np.where(ref_ch[1] > 0, ref_ch[1], np.nan)
        offsets.append(float(np.nanmedian((own_ch[0] - ref_ch[0]) / ref_std)))
    return offsets


def normalization_settings() -> dict[str, float]:
    """The preprocessing settings normalization statistics are only valid under.

    Statistics are per frequency bin, so they can only be reused with the same
    bins: same epoch length, band limits and notch.
    """
    return {
        "time_resolution_s": float(time_resolution),
        "low_cut_hz": LOW_CUT,
        "high_cut_hz": HIGH_CUT,
        "notch_low_cut_hz": NOTCH_LOW_CUT,
        "notch_high_cut_hz": NOTCH_HIGH_CUT,
        "trim_percent": ROBUST_TRIM_PERCENT,
    }


def incompatible_normalization_settings(metadata: dict[str, Any]) -> list[str]:
    """Names of the `normalization_settings()` saved metadata disagrees with (empty if none)."""
    return [
        name
        for name, value in normalization_settings().items()
        if not np.isclose(float(metadata.get(name, np.nan)), value)
    ]


def save_normalization_stats(
    path: str | Path, stats: ChannelStats, metadata: dict[str, Any] | None = None
) -> Path:
    """Write per-channel normalization statistics to an ``.npz`` file.

    `metadata` (JSON-serializable) is stored beside the arrays, together with
    `normalization_settings()`, so a reader can check the statistics are
    compatible before applying them to another recording.
    """
    path = Path(path)
    arrays: dict[str, np.ndarray] = {}
    for ch, entry in enumerate(stats):
        if entry is None:
            continue
        arrays[f"ch{ch}_mean"] = np.asarray(entry[0], dtype=float)
        arrays[f"ch{ch}_std"] = np.asarray(entry[1], dtype=float)
    header = {
        "format": NORMALIZATION_STATS_FORMAT,
        "n_channels": len(stats),
        **normalization_settings(),
        **(metadata or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        np.savez(handle, metadata=np.array(json.dumps(header, default=str)), **arrays)
    return path


def load_normalization_stats(path: str | Path) -> tuple[ChannelStats, dict[str, Any]]:
    """Read statistics written by `save_normalization_stats`: ``(stats, metadata)``."""
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata.get("format") != NORMALIZATION_STATS_FORMAT:
            raise ValueError(
                f"{path} is not a normalization statistics file "
                f"(format {metadata.get('format')!r})"
            )
        stats: ChannelStats = []
        for ch in range(int(metadata["n_channels"])):
            if f"ch{ch}_mean" in data:
                stats.append((data[f"ch{ch}_mean"].copy(), data[f"ch{ch}_std"].copy()))
            else:
                stats.append(None)
    return stats, metadata


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
    p: float = ROBUST_TRIM_PERCENT,
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


def resolve_normalization(
    prepared: PreparedRecording,
    sampling_rate_hz: float,
    *,
    global_normalization: bool = False,
    normalization_stats: StatsSource | None = None,
) -> NormalizationResult:
    """Decide which statistics a recording's scoring chunks are normalized against.

    `normalization_stats`, when it is (or, as a callable given `prepared`,
    returns) a stats list, wins -- source ``"reference"``. Otherwise
    `global_normalization` selects the recording's own pooled statistics
    (``"self"``), and failing that each chunk is normalized on its own
    (``"local"``). The recording's own pooled statistics are computed in
    every case and returned as `own`.
    """
    own = compute_global_normalization_stats(prepared, sampling_rate_hz)
    external = (
        normalization_stats(prepared) if callable(normalization_stats) else normalization_stats
    )
    if external is not None:
        source, applied = "reference", list(external)
    elif global_normalization:
        source, applied = "self", own
    else:
        source, applied = "local", None
    return NormalizationResult(
        source=source, applied=applied, own=own, signal_s=signal_duration_s(prepared)
    )

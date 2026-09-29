"""Detect and handle non-recorded periods in EEG recordings before scoring.

Disconnected or saturated EEG channels produce long runs of constant values.
If these are passed to Somnotate as-is, the robust z-score normalization in
preprocessing is computed over the constants (which dominate the percentile
trim once they exceed ~5% of the recording) and the rest of the recording is
normalized incorrectly. The result is degraded scoring across the entire
recording, not just the affected interval.

This module detects those intervals and prepares a recording for scoring using
one of three strategies, chosen automatically:

- ``trim_only``: missing data is only at the leading/trailing edges; trim it
  and score the contiguous middle.
- ``mask_inline``: small middle gaps (total <= ``max_missing_fraction`` and
  longest single gap <= ``max_single_gap_s``); score the whole kept range as
  one chunk and overwrite gap epochs as undefined in the output.
- ``split``: at least one middle gap is large; split the kept range at each
  gap boundary, score each contiguous chunk independently, fill gaps with
  undefined.

Callers can also pass ``exclude_intervals_s`` -- stretches known to be unusable
for another reason, e.g. long artifact periods from a pre-scoring scan. These
are handled exactly like detected gaps when choosing the strategy (trimmed,
masked or split around, and kept out of normalization), but are labelled
``artifact`` rather than ``gap`` so the two causes stay distinguishable.

All cut points are snapped to integer seconds (floor at start, ceil at end of
each detected gap or excluded interval) so the epoch grid is preserved
end-to-end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class RecordingSegment:
    """A non-overlapping slice of the original recording timeline."""

    segment_id: int
    kind: str  # "signal" | "gap" | "artifact" | "too_short"
    original_start_s: float
    original_end_s: float

    @property
    def duration_s(self) -> float:
        return self.original_end_s - self.original_start_s

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "kind": self.kind,
            "original_start_s": float(self.original_start_s),
            "original_end_s": float(self.original_end_s),
            "duration_s": float(self.duration_s),
        }


@dataclass
class ScoringChunk:
    """A contiguous block of raw signal that should be fed to the model.

    A single chunk may span several ``RecordingSegment`` entries (this happens
    in the ``mask_inline`` strategy, where one chunk covers the entire kept
    range and the middle gaps inside it are masked post-hoc).
    """

    original_start_s: float
    original_end_s: float
    raw_signal: np.ndarray  # (n_samples, n_channels)


@dataclass
class PreparedRecording:
    """Output of :func:`prepare_recording`."""

    segments: list[RecordingSegment]
    scoring_chunks: list[ScoringChunk]
    strategy: str  # "trim_only" | "mask_inline" | "split" | "all_missing"
    original_duration_s: float
    total_missing_s: float
    missing_fraction: float
    longest_gap_s: float
    sampling_rate_hz: float
    time_resolution_s: float

    def to_dict(self) -> dict:
        """JSON-serializable summary (no raw signal data)."""
        return {
            "strategy": self.strategy,
            "original_duration_s": float(self.original_duration_s),
            "total_missing_s": float(self.total_missing_s),
            "missing_fraction": float(self.missing_fraction),
            "longest_gap_s": float(self.longest_gap_s),
            "sampling_rate_hz": float(self.sampling_rate_hz),
            "time_resolution_s": float(self.time_resolution_s),
            "segments": [s.to_dict() for s in self.segments],
        }


def epoch_kinds(prepared: "PreparedRecording") -> np.ndarray:
    """Per-epoch ``kind`` ("signal" / "gap" / "artifact" / "too_short") for the entire original recording.

    Built from `prepared.segments`, at `prepared.time_resolution_s` resolution.
    Shared by scoring (to mask non-signal epochs out of the model output) and
    by global-normalization statistics (to exclude non-signal epochs from the
    pooled robust mean/std) so the two never disagree about which epochs are
    real signal.
    """
    time_res = prepared.time_resolution_s
    total_seconds = sum(s.duration_s for s in prepared.segments)
    n_epochs = int(round(total_seconds / time_res))

    kinds = np.empty(n_epochs, dtype=object)
    kinds[:] = "gap"
    for seg in prepared.segments:
        start_ep = int(round(seg.original_start_s / time_res))
        stop_ep = int(round(seg.original_end_s / time_res))
        kinds[start_ep:stop_ep] = seg.kind
    return kinds


def _detect_constant_runs(
    raw_signals: np.ndarray,
    missing_value_identifier: Optional[float] = None,
) -> np.ndarray:
    """Return a boolean mask of samples where any channel is in a constant run.

    A sample is marked constant when its local gradient and curvature are zero
    (the somnotate convention). If ``missing_value_identifier`` is supplied,
    we additionally require the value to equal that identifier exactly.
    """
    n_samples, n_channels = raw_signals.shape
    missing_any = np.zeros(n_samples, dtype=bool)
    for ch in range(n_channels):
        vec = raw_signals[:, ch]
        flag = np.zeros(n_samples, dtype=bool)
        if missing_value_identifier is not None:
            gradient = np.diff(vec)
            flag[:-1] = (gradient == 0) & (vec[:-1] == missing_value_identifier)
        elif n_samples >= 3:
            gradient = (vec[2:] - vec[:-2]) / 2.0
            curvature = np.diff(vec, 2)
            flag[1:-1] = (gradient == 0) & (curvature == 0)
        missing_any |= flag
    return missing_any


def _intervals_from_bool(mask: np.ndarray) -> np.ndarray:
    """Return an (N, 2) array of ``[start, stop)`` sample indices where mask is True."""
    if mask.size == 0:
        return np.zeros((0, 2), dtype=int)
    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    stops = np.where(diff == -1)[0]
    return np.stack([starts, stops], axis=1)


def _merge_overlapping(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, stop in intervals[1:]:
        last_start, last_stop = merged[-1]
        if start <= last_stop:
            merged[-1] = (last_start, max(last_stop, stop))
        else:
            merged.append((start, stop))
    return merged


def _missing_pieces(
    start_s: int, stop_s: int, excluded: list[tuple[int, int]]
) -> list[tuple[str, int, int]]:
    """Split the missing range ``[start_s, stop_s)`` into ``(kind, start, stop)`` pieces.

    Parts covered by ``excluded`` (sorted, non-overlapping) are ``artifact``;
    the rest is ``gap``.
    """
    pieces: list[tuple[str, int, int]] = []
    cursor = start_s
    for ex_start, ex_stop in excluded:
        ex_start, ex_stop = max(ex_start, start_s), min(ex_stop, stop_s)
        if ex_stop <= ex_start:
            continue
        if ex_start > cursor:
            pieces.append(("gap", cursor, ex_start))
        pieces.append(("artifact", ex_start, ex_stop))
        cursor = ex_stop
    if cursor < stop_s:
        pieces.append(("gap", cursor, stop_s))
    return pieces


def _all_missing(
    duration_int_s: int,
    original_duration_s: float,
    sampling_rate_hz: float,
    time_resolution_s: float,
    excluded: list[tuple[int, int]] | None = None,
) -> PreparedRecording:
    pieces = _missing_pieces(0, duration_int_s, excluded or []) or [
        ("gap", 0, duration_int_s)
    ]
    segments = [
        RecordingSegment(seg_id, kind, float(start), float(stop))
        for seg_id, (kind, start, stop) in enumerate(pieces)
    ]
    return PreparedRecording(
        segments=segments,
        scoring_chunks=[],
        strategy="all_missing",
        original_duration_s=original_duration_s,
        total_missing_s=float(duration_int_s),
        missing_fraction=1.0,
        longest_gap_s=float(duration_int_s),
        sampling_rate_hz=sampling_rate_hz,
        time_resolution_s=time_resolution_s,
    )


def prepare_recording(
    raw_signals: np.ndarray,
    sampling_rate_hz: float,
    *,
    time_resolution_s: float = 1.0,
    max_missing_fraction: float = 0.05,
    max_single_gap_s: float = 30 * 60,
    min_segment_length_s: float = 10 * 60,
    min_missing_run_s: float = 1.0,
    missing_value_identifier: Optional[float] = None,
    exclude_intervals_s: Optional[Sequence[tuple[float, float]]] = None,
) -> PreparedRecording:
    """Detect non-recorded periods and prepare a recording for scoring.

    Parameters
    ----------
    raw_signals
        ``(n_samples, n_channels)`` array as loaded from the EDF.
    sampling_rate_hz
        Sampling rate of the raw signals.
    time_resolution_s
        Epoch length in seconds (matches ``configuration.time_resolution``).
    max_missing_fraction
        Above this fraction of total kept duration spent in middle gaps, the
        strategy switches from ``mask_inline`` to ``split``. Defaulting to
        ``0.05`` matches the percentile-trim parameter ``p=5`` used by
        somnotate's ``robust_normalize``: gaps below this fraction are absorbed
        by the trim and don't bias normalization.
    max_single_gap_s
        Any single middle gap longer than this forces ``split`` regardless of
        the total fraction. Reason: even with normalization OK, the HMM forward-
        backward pass will run over the gap's garbage samples and can pull
        nearby real epochs around through state-transition smoothing.
    min_segment_length_s
        Signal chunks shorter than this (in the ``split`` strategy) are marked
        as ``too_short`` and not scored — too little context for the HMM.
    min_missing_run_s
        Constant-value runs shorter than this are ignored as not-really-missing
        (could be brief flat artefacts in real data).
    missing_value_identifier
        If supplied, only runs of *this exact value* are treated as missing.
        Otherwise any constant run qualifies.
    exclude_intervals_s
        Extra ``(start_s, end_s)`` intervals, in seconds from the start of the
        recording, to leave unscored -- e.g. long artifact periods. They count
        as missing data for the strategy decision and are labelled
        ``artifact`` (where they overlap a detected gap, ``artifact`` wins).

    Returns
    -------
    PreparedRecording
        Holds:

        - ``segments``: non-overlapping segments covering the *entire* original
          recording, classified as ``signal`` / ``gap`` / ``artifact`` /
          ``too_short``.
        - ``scoring_chunks``: the contiguous raw-signal blocks to feed the
          model (empty for ``all_missing``).
        - Strategy and summary statistics.

    Notes
    -----
    All cuts are snapped to integer-second boundaries. ``floor`` on the start of
    a detected gap and ``ceil`` on its end mean the cut is conservative: up to
    one second of clean data adjacent to each gap edge is discarded so the
    epoch grid stays aligned with the original recording's 0-second mark.
    """
    if raw_signals.ndim != 2:
        raise ValueError(
            f"raw_signals must be 2D (n_samples, n_channels); got shape {raw_signals.shape}"
        )

    n_samples = raw_signals.shape[0]
    original_duration_s = n_samples / float(sampling_rate_hz)
    duration_int_s = int(math.floor(original_duration_s))

    if duration_int_s <= 0:
        return _all_missing(0, original_duration_s, sampling_rate_hz, time_resolution_s)

    # 1. Detect missing samples (any channel constant).
    missing_mask = _detect_constant_runs(raw_signals, missing_value_identifier)
    missing_intervals = _intervals_from_bool(missing_mask)
    if len(missing_intervals) > 0:
        min_samples = int(math.ceil(min_missing_run_s * sampling_rate_hz))
        durations = missing_intervals[:, 1] - missing_intervals[:, 0]
        missing_intervals = missing_intervals[durations >= min_samples]

    # 2. Snap to integer seconds: floor start, ceil end. Clamp to recording bounds.
    snapped: list[tuple[int, int]] = []
    for start_sample, stop_sample in missing_intervals:
        start_s = max(0, int(math.floor(start_sample / sampling_rate_hz)))
        stop_s = min(duration_int_s, int(math.ceil(stop_sample / sampling_rate_hz)))
        if stop_s > start_s:
            snapped.append((start_s, stop_s))
    snapped = _merge_overlapping(snapped)

    # Caller-supplied exclusions, snapped the same way, count as missing too.
    excluded: list[tuple[int, int]] = []
    for start_s, stop_s in exclude_intervals_s or ():
        start_s = max(0, int(math.floor(start_s)))
        stop_s = min(duration_int_s, int(math.ceil(stop_s)))
        if stop_s > start_s:
            excluded.append((start_s, stop_s))
    excluded = _merge_overlapping(excluded)
    snapped = _merge_overlapping(snapped + excluded)

    # 3. Separate leading / trailing / middle.
    keep_start_s = 0
    keep_end_s = duration_int_s
    while snapped and snapped[0][0] <= keep_start_s:
        keep_start_s = max(keep_start_s, snapped[0][1])
        snapped.pop(0)
    while snapped and snapped[-1][1] >= keep_end_s:
        keep_end_s = min(keep_end_s, snapped[-1][0])
        snapped.pop()
    middle_gaps = snapped

    if keep_end_s <= keep_start_s:
        return _all_missing(
            duration_int_s,
            original_duration_s,
            sampling_rate_hz,
            time_resolution_s,
            excluded,
        )

    # 4. Decide strategy.
    kept_duration_s = keep_end_s - keep_start_s
    total_middle_missing_s = sum(stop - start for start, stop in middle_gaps)
    longest_middle_gap_s = max(
        (stop - start for start, stop in middle_gaps), default=0
    )
    middle_missing_fraction = (
        total_middle_missing_s / kept_duration_s if kept_duration_s > 0 else 0.0
    )

    if not middle_gaps:
        strategy = "trim_only"
    elif (
        middle_missing_fraction > max_missing_fraction
        or longest_middle_gap_s > max_single_gap_s
    ):
        strategy = "split"
    else:
        strategy = "mask_inline"

    # 5. Build segments and scoring_chunks.
    segments: list[RecordingSegment] = []
    scoring_chunks: list[ScoringChunk] = []
    seg_id = 0

    def _append_missing(start_s: int, stop_s: int) -> None:
        nonlocal seg_id
        for kind, piece_start, piece_stop in _missing_pieces(start_s, stop_s, excluded):
            segments.append(RecordingSegment(seg_id, kind, piece_start, piece_stop))
            seg_id += 1

    def _slice_signal(start_s: int, stop_s: int) -> np.ndarray:
        start_sample = int(round(start_s * sampling_rate_hz))
        stop_sample = int(round(stop_s * sampling_rate_hz))
        return raw_signals[start_sample:stop_sample]

    if keep_start_s > 0:
        _append_missing(0, keep_start_s)

    if strategy in ("trim_only", "mask_inline"):
        # The whole kept range is fed to the model as one chunk so the HMM gets
        # clean normalization. Middle gaps (in mask_inline) are recorded as
        # separate gap segments and will be overwritten as undefined post-hoc.
        cursor = keep_start_s
        for gap_start, gap_stop in middle_gaps:
            if gap_start > cursor:
                segments.append(
                    RecordingSegment(seg_id, "signal", cursor, gap_start)
                )
                seg_id += 1
            _append_missing(gap_start, gap_stop)
            cursor = gap_stop
        if cursor < keep_end_s:
            segments.append(RecordingSegment(seg_id, "signal", cursor, keep_end_s))
            seg_id += 1
        scoring_chunks.append(
            ScoringChunk(
                keep_start_s, keep_end_s, _slice_signal(keep_start_s, keep_end_s)
            )
        )
    else:  # split
        cursor = keep_start_s
        for gap_start, gap_stop in middle_gaps:
            sig_duration = gap_start - cursor
            if sig_duration > 0:
                kind = "signal" if sig_duration >= min_segment_length_s else "too_short"
                segments.append(RecordingSegment(seg_id, kind, cursor, gap_start))
                if kind == "signal":
                    scoring_chunks.append(
                        ScoringChunk(cursor, gap_start, _slice_signal(cursor, gap_start))
                    )
                seg_id += 1
            _append_missing(gap_start, gap_stop)
            cursor = gap_stop
        if cursor < keep_end_s:
            sig_duration = keep_end_s - cursor
            kind = "signal" if sig_duration >= min_segment_length_s else "too_short"
            segments.append(RecordingSegment(seg_id, kind, cursor, keep_end_s))
            if kind == "signal":
                scoring_chunks.append(
                    ScoringChunk(cursor, keep_end_s, _slice_signal(cursor, keep_end_s))
                )
            seg_id += 1

    if keep_end_s < duration_int_s:
        _append_missing(keep_end_s, duration_int_s)

    # 6. Stats over the entire original recording.
    total_missing_s = sum(
        s.duration_s for s in segments if s.kind in ("gap", "artifact", "too_short")
    )
    missing_fraction = total_missing_s / duration_int_s
    longest_gap_s = max(
        (s.duration_s for s in segments if s.kind in ("gap", "artifact")), default=0
    )

    return PreparedRecording(
        segments=segments,
        scoring_chunks=scoring_chunks,
        strategy=strategy,
        original_duration_s=original_duration_s,
        total_missing_s=total_missing_s,
        missing_fraction=missing_fraction,
        longest_gap_s=longest_gap_s,
        sampling_rate_hz=sampling_rate_hz,
        time_resolution_s=time_resolution_s,
    )

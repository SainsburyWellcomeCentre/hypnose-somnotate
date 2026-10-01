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
- ``mask_inline``: every middle gap is short (<= ``max_single_gap_s``); score
  the whole kept range as one chunk and overwrite gap epochs as undefined in
  the output.
- ``split``: at least one middle gap is long (> ``max_single_gap_s``); split the
  kept range at each long gap only, score each chunk independently, and mask
  the short gaps inside each chunk as in ``mask_inline``.

Under every strategy, a chunk shorter than ``min_segment_length_s`` is not
scored -- too little context for the HMM -- and its signal is labelled
``too_short``. A recording whose kept range is itself that short is therefore
left entirely unscored.

The decision is made per gap, by duration alone -- there is no total-missing-
fraction trigger, since short zero-filled gaps are kept out of normalization by
global normalization (``preprocessing.compute_global_normalization_stats``)
rather than by splitting around them.

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

# Middle gaps longer than this split a recording into separately scored chunks.
DEFAULT_MAX_SINGLE_GAP_S = 30 * 60
# Chunks shorter than this are left unscored (``too_short``).
DEFAULT_MIN_SEGMENT_LENGTH_S = 10 * 60


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
    in the ``mask_inline`` and ``split`` strategies, where short middle gaps
    inside a chunk are masked post-hoc rather than split around).
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
    max_single_gap_s: float = DEFAULT_MAX_SINGLE_GAP_S,
    min_segment_length_s: float = DEFAULT_MIN_SEGMENT_LENGTH_S,
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
    max_single_gap_s
        Middle gaps longer than this split the recording into separately scored
        chunks; shorter ones are scored through and masked as undefined.
        Reason: the HMM forward-backward pass runs over a masked gap's garbage
        samples and can pull nearby real epochs around through state-transition
        smoothing, which matters more the longer the gap.
    min_segment_length_s
        Chunks shorter than this are not scored, under every strategy -- too
        little context for the HMM; their signal is marked ``too_short``. A
        short recording (or one whose kept range is short) is a single short
        chunk, so it is left unscored entirely.
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

    # 4. Decide strategy: only middle gaps longer than max_single_gap_s split.
    split_gaps = [
        (start, stop) for start, stop in middle_gaps if stop - start > max_single_gap_s
    ]
    if not middle_gaps:
        strategy = "trim_only"
    elif split_gaps:
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

    # Chunk boundaries are the long gaps. Short gaps inside a chunk are recorded
    # as separate gap segments and overwritten as undefined post-hoc, while the
    # chunk itself is fed to the model whole.
    chunk_ranges: list[tuple[int, int]] = []
    cursor = keep_start_s
    for gap_start, gap_stop in split_gaps:
        chunk_ranges.append((cursor, gap_start))
        cursor = gap_stop
    chunk_ranges.append((cursor, keep_end_s))

    for chunk_index, (chunk_start, chunk_stop) in enumerate(chunk_ranges):
        if chunk_index > 0:
            _append_missing(*split_gaps[chunk_index - 1])
        scored = chunk_stop - chunk_start >= min_segment_length_s
        signal_kind = "signal" if scored else "too_short"
        cursor = chunk_start
        for gap_start, gap_stop in middle_gaps:
            if gap_start < chunk_start or gap_stop > chunk_stop:
                continue
            if gap_start > cursor:
                segments.append(RecordingSegment(seg_id, signal_kind, cursor, gap_start))
                seg_id += 1
            _append_missing(gap_start, gap_stop)
            cursor = gap_stop
        if cursor < chunk_stop:
            segments.append(RecordingSegment(seg_id, signal_kind, cursor, chunk_stop))
            seg_id += 1
        if scored:
            scoring_chunks.append(
                ScoringChunk(chunk_start, chunk_stop, _slice_signal(chunk_start, chunk_stop))
            )

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

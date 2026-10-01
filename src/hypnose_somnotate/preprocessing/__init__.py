"""Signal preprocessing and recording gap correction."""

from .gap_correction import (
    PreparedRecording,
    RecordingSegment,
    ScoringChunk,
    prepare_recording,
)
from .preprocessing import (
    NormalizationResult,
    load_normalization_stats,
    normalization_offset_z,
    preprocess_multichannel,
    save_normalization_stats,
    signal_duration_s,
)

__all__ = [
    "NormalizationResult",
    "PreparedRecording",
    "RecordingSegment",
    "ScoringChunk",
    "load_normalization_stats",
    "normalization_offset_z",
    "prepare_recording",
    "preprocess_multichannel",
    "save_normalization_stats",
    "signal_duration_s",
]

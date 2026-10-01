"""Reference-baseline normalization: resolving, saving and comparing statistics."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hypnose_somnotate.preprocessing.gap_correction import (
    PreparedRecording,
    RecordingSegment,
    ScoringChunk,
)
from hypnose_somnotate.preprocessing.preprocessing import (
    compute_global_normalization_stats,
    incompatible_normalization_settings,
    load_normalization_stats,
    normalization_offset_z,
    normalization_settings,
    resolve_normalization,
    save_normalization_stats,
    signal_duration_s,
)

SAMPLING_RATE_HZ = 256.0


def _prepared(duration_s: int = 120, *, scale: float = 1.0, seed: int = 0) -> PreparedRecording:
    """A single-chunk recording of white noise, `duration_s` seconds of signal."""
    rng = np.random.default_rng(seed)
    n_samples = int(duration_s * SAMPLING_RATE_HZ)
    signal = scale * rng.standard_normal((n_samples, 3))
    return PreparedRecording(
        segments=[RecordingSegment(0, "signal", 0.0, float(duration_s))],
        scoring_chunks=[ScoringChunk(0.0, float(duration_s), signal)],
        strategy="trim_only",
        original_duration_s=float(duration_s),
        total_missing_s=0.0,
        missing_fraction=0.0,
        longest_gap_s=0.0,
        sampling_rate_hz=SAMPLING_RATE_HZ,
        time_resolution_s=1.0,
    )


def test_signal_duration_counts_only_signal_segments() -> None:
    prepared = _prepared(60)
    prepared.segments = [
        RecordingSegment(0, "signal", 0.0, 40.0),
        RecordingSegment(1, "artifact", 40.0, 50.0),
        RecordingSegment(2, "signal", 50.0, 60.0),
    ]
    assert signal_duration_s(prepared) == 50.0


def test_resolve_normalization_prefers_supplied_statistics() -> None:
    prepared = _prepared()
    reference = compute_global_normalization_stats(_prepared(seed=1), SAMPLING_RATE_HZ)

    result = resolve_normalization(
        prepared, SAMPLING_RATE_HZ, global_normalization=True, normalization_stats=reference
    )

    assert result.source == "reference"
    assert result.applied is not None
    assert np.array_equal(result.applied[0][0], reference[0][0])
    assert not np.array_equal(result.own[0][0], reference[0][0])
    assert result.signal_s == 120.0


def test_resolve_normalization_callable_returning_none_falls_back() -> None:
    prepared = _prepared()
    seen = []

    def resolver(candidate, own):
        seen.append(candidate)
        assert len(own) == 3
        return None

    result = resolve_normalization(
        prepared, SAMPLING_RATE_HZ, global_normalization=True, normalization_stats=resolver
    )
    assert seen == [prepared]
    assert result.source == "self"
    assert result.applied is result.own

    local = resolve_normalization(prepared, SAMPLING_RATE_HZ, normalization_stats=resolver)
    assert local.source == "local"
    assert local.applied is None


def test_saved_statistics_round_trip(tmp_path) -> None:
    stats = compute_global_normalization_stats(_prepared(), SAMPLING_RATE_HZ)
    stats[1] = None
    path = save_normalization_stats(
        tmp_path / "rec_somnotate_normalization.npz", stats, {"signal_s": 120.0, "edf_name": "rec.edf"}
    )

    loaded, metadata = load_normalization_stats(path)

    assert metadata["signal_s"] == 120.0
    assert metadata["edf_name"] == "rec.edf"
    assert incompatible_normalization_settings(metadata) == []
    assert loaded[1] is None
    for original, restored in ((stats[0], loaded[0]), (stats[2], loaded[2])):
        assert np.allclose(original[0], restored[0])
        assert np.allclose(original[1], restored[1])


def test_settings_mismatch_is_reported() -> None:
    metadata = {**normalization_settings(), "high_cut_hz": 45.0}
    assert incompatible_normalization_settings(metadata) == ["high_cut_hz"]


def test_load_rejects_other_npz_files(tmp_path) -> None:
    path = tmp_path / "other.npz"
    np.savez(path, metadata=np.array(json.dumps({"format": "something-else"})))
    with pytest.raises(ValueError, match="not a normalization statistics file"):
        load_normalization_stats(path)


def test_offset_is_near_zero_for_matching_recordings_and_large_for_a_gain_change() -> None:
    reference = compute_global_normalization_stats(_prepared(seed=1), SAMPLING_RATE_HZ)
    same = compute_global_normalization_stats(_prepared(seed=2), SAMPLING_RATE_HZ)
    amplified = compute_global_normalization_stats(_prepared(seed=2, scale=4.0), SAMPLING_RATE_HZ)

    same_offsets = normalization_offset_z(same, reference)
    gain_offsets = normalization_offset_z(amplified, reference)

    assert all(abs(offset) < 0.5 for offset in same_offsets)
    assert all(offset > 1.0 for offset in gain_offsets)
    assert normalization_offset_z([None, same[1]], reference)[0] is None


class _StubHMM:
    states: list = []

    def predict_proba(self, samples):
        return np.zeros((len(samples), 0))


class _StubAnnotator:
    """Records the preprocessed features it is asked to score."""

    def __init__(self) -> None:
        self.hmm = _StubHMM()
        self.inputs: list[np.ndarray] = []

    def predict(self, features):
        self.inputs.append(features)
        return np.ones(len(features), dtype=int)

    def transform(self, features):
        return features


def test_scoring_normalizes_chunks_against_the_supplied_reference() -> None:
    from hypnose_somnotate.scoring.scoring import _score_prepared_recording

    reference = compute_global_normalization_stats(_prepared(seed=1, scale=4.0), SAMPLING_RATE_HZ)

    own_annotator, ref_annotator = _StubAnnotator(), _StubAnnotator()
    own_prepared, ref_prepared = _prepared(), _prepared()
    _score_prepared_recording(
        own_prepared, own_annotator, SAMPLING_RATE_HZ, global_normalization=True
    )
    df = _score_prepared_recording(
        ref_prepared, ref_annotator, SAMPLING_RATE_HZ,
        global_normalization=True, normalization_stats=lambda prepared, own: reference,
    )

    assert own_prepared.normalization.source == "self"
    assert ref_prepared.normalization.source == "reference"
    assert ref_prepared.to_dict()["normalization"] == {"source": "reference", "signal_s": 120.0}
    # Normalized against its own stats the features are centred; against a
    # 4x-gain reference they sit well below it.
    assert abs(float(np.median(own_annotator.inputs[0]))) < 0.5
    assert float(np.median(ref_annotator.inputs[0])) < -1.0
    assert len(df) == 120

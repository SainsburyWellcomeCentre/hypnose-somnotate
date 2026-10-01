"""Scoring workflow for Somnotate models."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from ..somnotate._automated_state_annotation import StateAnnotator
from ..somnotate._utils import convert_state_vector_to_state_intervals
from ..somnotate_pipeline.utils.configuration import time_resolution
from ..somnotate_pipeline.io.data_io import load_raw_signals, export_hypnogram

from ..config import (
    DEFAULT_CHANNEL_LABELS,
    DEFAULT_SAMPLING_RATE_HZ,
    MODEL_TO_OUTPUT_LABEL,
    PROBABILITY_JSON_KEYS,
)
from ..preprocessing.gap_correction import (
    DEFAULT_MAX_SINGLE_GAP_S,
    DEFAULT_MIN_SEGMENT_LENGTH_S,
    PreparedRecording,
    epoch_kinds,
    prepare_recording,
)
from ..io.loading import (
    hypnogram_path,
    normalization_stats_path,
    prediction_path,
    segments_path,
)
from ..io.paths import find_recordings, get_derivatives_root
from ..preprocessing.preprocessing import (
    ChannelStats,
    StatsSource,
    compute_global_normalization_stats,
    preprocess_multichannel,
    resolve_normalization,
    save_normalization_stats,
)
from ..somnotate_pipeline.utils import configuration


UNDEFINED_MODEL_LABEL = 0
UNDEFINED_OUTPUT_LABEL = MODEL_TO_OUTPUT_LABEL.get(UNDEFINED_MODEL_LABEL, 3)


def _load_annotator(model_path: Path) -> StateAnnotator:
    annotator = StateAnnotator()
    annotator.load(str(model_path))
    return annotator


def score_recording(
    edf_path: Path,
    model: Path | StateAnnotator,
    *,
    channel_labels: list[str] | None = None,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
    global_normalization: bool = False,
    normalization_stats: StatsSource | None = None,
    exclude_intervals_s: list[tuple[float, float]] | None = None,
    max_single_gap_s: float = DEFAULT_MAX_SINGLE_GAP_S,
    min_segment_length_s: float = DEFAULT_MIN_SEGMENT_LENGTH_S,
) -> tuple[pd.DataFrame, PreparedRecording]:
    """Score one EDF recording, given only the file itself and a trained model.

    This is the layout-unaware core of the scoring pipeline: no subject/session
    directory conventions, no derivatives root, no opinion about which of
    several files in a folder should be scored -- just "here is one EDF, here
    is a model, give me predictions for it". `score_recordings` (below) is a
    batch wrapper built on top of this that adds exactly those layout
    decisions via `io.paths.find_recordings`; a caller with its own
    file-discovery and concatenation-preference logic -- e.g.
    hypnose-eeg-analysis's `scripts/sleep_scoring/score_recordings.py` -- can
    call this directly instead and own those decisions itself.

    model
        Either a path to a trained ``model.pickle``, or an already-loaded
        `StateAnnotator`. Pass a loaded annotator when scoring many
        recordings with the same model, to load the pickle once rather than
        once per call.
    normalization_stats
        Optional per-channel ``(robust_mean, robust_std)`` statistics to
        normalize every chunk against instead of the recording's own --
        typically pooled from a longer recording of the same animal (see
        ``recording_normalization_stats``), for recordings too short to
        provide a representative baseline themselves. May also be a callable
        taking the `PreparedRecording` and the recording's own pooled
        statistics and returning such a list, or None to fall back to
        `global_normalization` -- so the choice can depend on how much signal
        the recording turns out to have and how far it sits from a candidate
        reference. Which statistics
        were used is recorded on the returned ``prepared.normalization``.
    exclude_intervals_s
        Optional ``(start_s, end_s)`` intervals, in seconds from the start of
        the recording, to leave unscored (e.g. long artifact periods). They are
        handled like detected gaps -- trimmed, masked or split around, and kept
        out of normalization -- and labelled undefined with kind ``artifact``.
    max_single_gap_s
        Middle gaps (including excluded intervals) longer than this split the
        recording into separately scored chunks; shorter ones are scored
        through and masked as undefined. See
        ``preprocessing.gap_correction.prepare_recording``.
    min_segment_length_s
        Chunks (or whole recordings) shorter than this are left unscored and
        labelled undefined with kind ``too_short`` -- too little context for
        the HMM.

    Returns
    -------
    (predictions_df, prepared)
        The same per-epoch DataFrame and `PreparedRecording` gap/chunk plan
        that `score_recordings` writes to the predictions parquet and
        segments JSON, respectively.
    """
    channel_labels = channel_labels or DEFAULT_CHANNEL_LABELS
    annotator = model if isinstance(model, StateAnnotator) else _load_annotator(model)

    raw_signals = load_raw_signals(str(edf_path), channel_labels)
    prepared = prepare_recording(
        raw_signals,
        sampling_rate_hz=sampling_rate_hz,
        time_resolution_s=time_resolution,
        exclude_intervals_s=exclude_intervals_s,
        max_single_gap_s=max_single_gap_s,
        min_segment_length_s=min_segment_length_s,
    )
    df = _score_prepared_recording(
        prepared,
        annotator,
        sampling_rate_hz=sampling_rate_hz,
        global_normalization=global_normalization,
        normalization_stats=normalization_stats,
    )
    return df, prepared


def recording_normalization_stats(
    edf_path: Path,
    *,
    channel_labels: list[str] | None = None,
    sampling_rate_hz: float = DEFAULT_SAMPLING_RATE_HZ,
    exclude_intervals_s: list[tuple[float, float]] | None = None,
    max_single_gap_s: float = DEFAULT_MAX_SINGLE_GAP_S,
    min_segment_length_s: float = DEFAULT_MIN_SEGMENT_LENGTH_S,
) -> tuple[ChannelStats, PreparedRecording]:
    """A recording's own pooled normalization statistics, without scoring it.

    The same statistics `score_recording` computes as ``prepared.normalization.own``
    -- signal epochs only, after the same gap/artifact handling -- for a
    recording scored before those were saved, or not scored at all. Pass the
    same `exclude_intervals_s` and gap settings its scoring uses (or would use).
    """
    channel_labels = channel_labels or DEFAULT_CHANNEL_LABELS
    raw_signals = load_raw_signals(str(edf_path), channel_labels)
    prepared = prepare_recording(
        raw_signals,
        sampling_rate_hz=sampling_rate_hz,
        time_resolution_s=time_resolution,
        exclude_intervals_s=exclude_intervals_s,
        max_single_gap_s=max_single_gap_s,
        min_segment_length_s=min_segment_length_s,
    )
    return compute_global_normalization_stats(prepared, sampling_rate_hz), prepared


def score_recordings(
    subjids: list[int | str],
    model_path: Path,
    repo_root: Path,
    dates: list[int | str] | None = None,
    date_range: tuple[int | str, int | str] | None = None,
    channel_labels: list[str] | None = None,
    export_visbrain: bool = True,
    sampling_rate_hz: int = DEFAULT_SAMPLING_RATE_HZ,
    output_subdir: str = "saved_results",
    global_normalization: bool = False,
    max_single_gap_s: float = DEFAULT_MAX_SINGLE_GAP_S,
    min_segment_length_s: float = DEFAULT_MIN_SEGMENT_LENGTH_S,
) -> list[Path]:
    derivatives_root = get_derivatives_root(repo_root)
    if not derivatives_root.exists():
        raise FileNotFoundError(
            f"Derivatives root not found at {derivatives_root}. Create a symlink named 'derivatives' in the repo."
        )

    channel_labels = channel_labels or DEFAULT_CHANNEL_LABELS
    recordings = find_recordings(
        repo_root, subjids, dates=dates, date_range=date_range, output_subdir=output_subdir
    )

    annotator = _load_annotator(model_path)

    output_paths: list[Path] = []
    for recording in recordings:
        df, prepared = score_recording(
            recording.edf_path,
            annotator,
            channel_labels=channel_labels,
            sampling_rate_hz=sampling_rate_hz,
            global_normalization=global_normalization,
            max_single_gap_s=max_single_gap_s,
            min_segment_length_s=min_segment_length_s,
        )
        _print_recording_plan(recording, prepared)

        output_dir = recording.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        # Output naming is owned by io.loading, so the readers there and this writer
        # cannot drift apart.
        output_path = prediction_path(recording)
        sidecar_path = segments_path(recording)

        df.to_parquet(output_path, index=False)
        with open(sidecar_path, "w") as f:
            json.dump(prepared.to_dict(), f, indent=2)
        if prepared.normalization is not None and prepared.normalization.own:
            save_normalization_stats(
                normalization_stats_path(recording),
                prepared.normalization.own,
                {
                    "edf_name": recording.edf_path.name,
                    "channel_labels": channel_labels,
                    "sampling_rate_hz": float(sampling_rate_hz),
                    "signal_s": prepared.normalization.signal_s,
                },
            )

        if export_visbrain:
            hyp_path = hypnogram_path(recording)
            states, intervals = convert_state_vector_to_state_intervals(
                df["label_model"].to_numpy(dtype=int),
                mapping=configuration.int_to_state,
                time_resolution=time_resolution,
            )
            export_hypnogram(str(hyp_path), states, intervals)

        output_paths.append(output_path)

    return output_paths


def _score_prepared_recording(
    prepared: PreparedRecording,
    annotator: StateAnnotator,
    sampling_rate_hz: float,
    *,
    global_normalization: bool = False,
    normalization_stats: StatsSource | None = None,
) -> pd.DataFrame:
    """Run the model on each scoring chunk and assemble a per-epoch DataFrame.

    Output covers the entire original recording in epoch time. Epochs that fall
    in ``gap``, ``artifact`` or ``too_short`` segments are filled with the undefined label and
    zero probabilities.

    global_normalization
        If True, every chunk is normalized against one set of robust
        mean/std statistics pooled across all of ``prepared.scoring_chunks``
        (gap/artifact/too_short epochs excluded) instead of each chunk's own
        statistics -- see
        ``preprocessing.preprocessing.compute_global_normalization_stats``.
        Matters most for the ``split`` strategy, where several independent
        chunks would otherwise each get their own baseline.
    normalization_stats
        Statistics (or a callable choosing them) that take precedence over
        both of the above -- see `score_recording`. The decision, and the
        recording's own pooled statistics, are stored on
        ``prepared.normalization``.
    """
    time_res = prepared.time_resolution_s
    total_seconds = sum(s.duration_s for s in prepared.segments)
    n_epochs = int(round(total_seconds / time_res))

    label_model = np.full(n_epochs, UNDEFINED_MODEL_LABEL, dtype=int)
    label_output = np.full(n_epochs, UNDEFINED_OUTPUT_LABEL, dtype=int)
    segment_ids = np.full(n_epochs, -1, dtype=int)
    kinds = epoch_kinds(prepared)

    prob_columns: dict[str, np.ndarray] = {
        "prob_wake": np.zeros(n_epochs, dtype=float),
        "prob_nrem": np.zeros(n_epochs, dtype=float),
        "prob_rem": np.zeros(n_epochs, dtype=float),
        "prob_undef": np.zeros(n_epochs, dtype=float),
    }
    prob_key_to_column = {
        "W": "prob_wake",
        "N": "prob_nrem",
        "R": "prob_rem",
        "U": "prob_undef",
    }

    for seg in prepared.segments:
        start_ep = int(round(seg.original_start_s / time_res))
        stop_ep = int(round(seg.original_end_s / time_res))
        segment_ids[start_ep:stop_ep] = seg.segment_id

    normalization = resolve_normalization(
        prepared,
        sampling_rate_hz,
        global_normalization=global_normalization,
        normalization_stats=normalization_stats,
    )
    prepared.normalization = normalization

    for chunk in prepared.scoring_chunks:
        preprocessed = preprocess_multichannel(
            chunk.raw_signal, sampling_rate_hz, normalization_stats=normalization.applied
        )
        chunk_predicted = np.abs(np.asarray(annotator.predict(preprocessed), dtype=int))
        chunk_probs = _predict_state_probabilities(annotator, preprocessed)

        chunk_start_ep = int(round(chunk.original_start_s / time_res))
        chunk_n = len(chunk_predicted)
        chunk_stop_ep = chunk_start_ep + chunk_n

        chunk_kinds = kinds[chunk_start_ep:chunk_stop_ep]
        signal_mask = chunk_kinds == "signal"
        if not np.any(signal_mask):
            continue

        chunk_output = np.fromiter(
            (MODEL_TO_OUTPUT_LABEL.get(int(v), UNDEFINED_OUTPUT_LABEL) for v in chunk_predicted),
            dtype=int,
            count=chunk_n,
        )

        label_model_view = label_model[chunk_start_ep:chunk_stop_ep]
        label_output_view = label_output[chunk_start_ep:chunk_stop_ep]
        label_model_view[signal_mask] = chunk_predicted[signal_mask]
        label_output_view[signal_mask] = chunk_output[signal_mask]
        label_model[chunk_start_ep:chunk_stop_ep] = label_model_view
        label_output[chunk_start_ep:chunk_stop_ep] = label_output_view

        for prob_key, state_int in PROBABILITY_JSON_KEYS.items():
            column = prob_key_to_column.get(prob_key)
            if column is None:
                continue
            chunk_vec = chunk_probs.get(state_int, np.zeros(chunk_n))
            chunk_vec = np.clip(chunk_vec.astype(float), 0.0, 1.0)
            prob_view = prob_columns[column][chunk_start_ep:chunk_stop_ep]
            prob_view[signal_mask] = chunk_vec[signal_mask]
            prob_columns[column][chunk_start_ep:chunk_stop_ep] = prob_view

    timepoints = np.arange(n_epochs, dtype=float) * time_res

    df = pd.DataFrame(
        {
            "time_s": timepoints,
            "label": label_output,
            "label_model": label_model,
            "label_output": label_output,
            "segment_id": segment_ids,
            "kind": kinds.astype(str),
            **prob_columns,
        }
    )
    return df


_STRATEGY_DESCRIPTIONS = {
    "trim_only": "trim leading/trailing gaps, score the contiguous middle",
    "mask_inline": "score the kept range as one chunk, mark middle gaps as undefined post-hoc",
    "split": "split into independent chunks at long middle gaps, mark all gaps as undefined",
    "all_missing": "no scorable signal in this recording (skipping)",
}


_NORMALIZATION_DESCRIPTIONS = {
    "reference": "statistics supplied by the caller (e.g. a longer recording of the same animal)",
    "self": "pooled across all scoring chunks, gap/artifact/too_short epochs excluded",
    "local": "each scoring chunk against its own statistics",
}


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def _print_recording_plan(recording, prepared: PreparedRecording) -> None:
    strategy = prepared.strategy
    strategy_desc = _STRATEGY_DESCRIPTIONS.get(strategy, "")
    gap_segs = [s for s in prepared.segments if s.kind == "gap"]
    artifact_segs = [s for s in prepared.segments if s.kind == "artifact"]
    too_short_segs = [s for s in prepared.segments if s.kind == "too_short"]

    print(f"[{recording.subject} {recording.session} date-{recording.date}] {recording.edf_path.name}")
    print(
        f"  Duration: {_format_duration(prepared.original_duration_s)} | "
        f"missing: {_format_duration(prepared.total_missing_s)} "
        f"({100 * prepared.missing_fraction:.1f}%) | "
        f"longest gap: {_format_duration(prepared.longest_gap_s)}"
    )
    print(f"  Strategy: {strategy} — {strategy_desc}")

    if not prepared.segments:
        print("  (no segments)")
        return

    print(f"  Segments ({len(prepared.segments)}):")
    for seg in prepared.segments:
        start = int(round(seg.original_start_s))
        stop = int(round(seg.original_end_s))
        marker = ""
        if seg.kind == "signal":
            chunk_idx = next(
                (
                    idx + 1
                    for idx, c in enumerate(prepared.scoring_chunks)
                    if c.original_start_s <= seg.original_start_s
                    and seg.original_end_s <= c.original_end_s
                ),
                None,
            )
            if chunk_idx is not None:
                marker = f"  -> chunk {chunk_idx}"
        elif seg.kind in ("artifact", "too_short"):
            marker = "  (not scored)"
        print(
            f"    [{start:>8d} -> {stop:>8d} s]  {seg.kind:<10s} "
            f"({_format_duration(seg.duration_s)}){marker}"
        )

    if too_short_segs:
        print(
            f"  Note: {len(too_short_segs)} segment(s) marked too_short "
            "(below min_segment_length_s) and will be filled as undefined."
        )
    if artifact_segs:
        print(
            f"  Note: {len(artifact_segs)} segment(s) excluded as artifact "
            "and will be filled as undefined."
        )
    if not gap_segs and not artifact_segs and strategy != "all_missing":
        print("  No gaps detected.")
    if prepared.normalization is not None:
        description = _NORMALIZATION_DESCRIPTIONS.get(prepared.normalization.source, "")
        print(f"  Normalization: {prepared.normalization.source} — {description}")


def _predict_state_probabilities(annotator: StateAnnotator, signal_array: np.ndarray) -> dict[int, np.ndarray]:
    transformed = annotator.transform(signal_array)
    probability_array = annotator.hmm.predict_proba([sample for sample in transformed])

    probability_dict: dict[int, np.ndarray] = {}
    for ii, state in enumerate(annotator.hmm.states):
        if state.distribution is None:
            continue
        probability_dict[int(state.name)] = probability_array[:, ii]

    return probability_dict

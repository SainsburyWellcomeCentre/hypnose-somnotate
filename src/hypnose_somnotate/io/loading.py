"""Readers for pipeline artifacts on disk.

Everything here is I/O: resolving derivative paths, iterating directories, and
reading/parsing the files the pipeline writes. Nothing in this module runs a
model or produces a figure, and it imports only from `.paths` and `..config`,
so it stays cheap and free of cycles with the analysis modules that use it.

`load_scored_recording` is the selector-driven entry point: subject + date in,
loaded arrays out.
"""

from __future__ import annotations

import json
import re
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..config import (
    DEFAULT_CHANNEL_LABELS,
    DEFAULT_RECORDING_INDEX,
    DEFAULT_SAMPLING_RATE_HZ,
    MODEL_TO_OUTPUT_LABEL,
)
from .paths import find_recordings, get_derivatives_root
from .selectors import parse_date_range, parse_dates, parse_subjects


# --------------------------------------------------------------------------------
# Derivative path resolution
# --------------------------------------------------------------------------------

def get_test_root(repo_root: Path, model_name: str, test_name: str | None) -> Path:
    root = get_derivatives_root(repo_root) / "somnotate_testing" / model_name
    return root / test_name if test_name else root


def get_csv_dir(repo_root: Path, model_name: str, test_name: str | None) -> Path:
    return get_test_root(repo_root, model_name, test_name) / "intermediate" / "csv"


def get_predictions_dir(repo_root: Path, model_name: str, test_name: str | None) -> Path:
    return get_test_root(repo_root, model_name, test_name) / "somnotate_predictions"


# --------------------------------------------------------------------------------
# Signal tables (parquet / CSV)
# --------------------------------------------------------------------------------

SIGNAL_TABLE_SUFFIXES: tuple[str, ...] = (".parquet", ".csv")


def load_signal_table(path: Path) -> pd.DataFrame:
    """Read a per-recording signal table (parquet or CSV) into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, encoding="latin1")
    raise ValueError(f"Unsupported signal-table format: {path}")


def list_csv_files(csv_dir: Path) -> list[Path]:
    """Return signal-table files in csv_dir (parquet + csv).

    If both a .parquet and .csv exist for the same stem, parquet wins.
    Kept under the legacy name `list_csv_files` for API stability.
    """
    if not csv_dir.exists():
        return []
    by_stem: dict[str, Path] = {}
    # Iterate in order of suffix preference: parquet first, csv as fallback.
    for suffix in SIGNAL_TABLE_SUFFIXES:
        for path in csv_dir.glob(f"*{suffix}"):
            if path.name.startswith("._"):
                continue
            by_stem.setdefault(path.stem, path)
    return sorted(by_stem.values())


def find_somnotate_predictions(predictions_dir: Path) -> Path | None:
    if not predictions_dir.exists():
        return None
    preds = sorted(p for p in predictions_dir.glob("*_somnotate.parquet") if not p.name.startswith("._"))
    return preds[0] if preds else None


# --------------------------------------------------------------------------------
# Prediction readers
# --------------------------------------------------------------------------------

def load_somnotate_predictions(pred_path: Path) -> np.ndarray:
    if pred_path.suffix == ".parquet":
        df = pd.read_parquet(pred_path)
    else:
        df = pd.read_csv(pred_path)
    if "label" not in df.columns:
        raise ValueError(f"Missing 'label' column in {pred_path}")
    return df["label"].to_numpy(dtype=int)


def load_somnotate_vector(pred_path: Path) -> np.ndarray:
    """Read a scoring predictions parquet into a model-label vector.

    Prefers the `label_model` column; falls back to inverting MODEL_TO_OUTPUT_LABEL
    on `label` for older prediction files that only stored output labels.
    """
    pred_df = pd.read_parquet(pred_path)
    if "label_model" in pred_df.columns:
        return pred_df["label_model"].to_numpy(dtype=int)
    if "label" not in pred_df.columns:
        raise ValueError(f"No 'label_model' or 'label' column in {pred_path}")
    inverse_map = {value: key for key, value in MODEL_TO_OUTPUT_LABEL.items()}
    return pred_df["label"].map(lambda v: inverse_map.get(int(v), 0)).to_numpy(dtype=int)


# --------------------------------------------------------------------------------
# Manually annotated testing tables
# --------------------------------------------------------------------------------

def _normalize_stage_values(stage_values: Iterable) -> np.ndarray:
    series = pd.to_numeric(pd.Series(stage_values), errors="coerce").fillna(0).astype(int)
    stage_map = {1: 1, 2: 2, 3: 3, 4: 0, 0: 0}
    mapped = series.map(lambda v: stage_map.get(int(v), 0)).to_numpy(dtype=int)
    return mapped


def _collapse_stage_vector(
    stage_values: np.ndarray,
    sampling_rate_hz: int,
    time_resolution_s: float,
) -> np.ndarray:
    epoch_samples = int(round(sampling_rate_hz * time_resolution_s))
    if epoch_samples <= 0:
        raise ValueError("epoch_samples must be >= 1")

    total_epochs = len(stage_values) // epoch_samples
    if total_epochs == 0:
        return np.array([], dtype=int)

    trimmed = stage_values[: total_epochs * epoch_samples]
    reshaped = trimmed.reshape(total_epochs, epoch_samples)
    epoch_labels = reshaped[:, 0]

    non_uniform = np.sum(~np.all(reshaped == reshaped[:, [0]], axis=1))
    if non_uniform:
        print(
            "Warning: sleepStage values vary within some epochs; "
            f"using the first sample for {non_uniform} epochs."
        )

    return epoch_labels.astype(int)


def load_testing_csv(
    csv_path: Path,
    sampling_rate_hz: int = DEFAULT_SAMPLING_RATE_HZ,
    time_resolution_s: float | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if csv_path.name.startswith("._"):
        raise ValueError(f"Metadata file not supported: {csv_path}")

    # `if not …` (not `is None`) to preserve the original `x = x or default` semantics
    if not time_resolution_s:
        # imported lazily: configuration pulls in matplotlib and mutates rcParams
        from ..somnotate_pipeline.utils import configuration

        time_resolution_s = configuration.time_resolution
    df = load_signal_table(csv_path)

    required = ["EEG1", "EEG2", "EMG"]
    if not set(required).issubset(df.columns):
        raise ValueError(f"Missing required columns in {csv_path}")

    raw_signals = df[required].to_numpy(dtype=float)

    stage_columns = [col for col in df.columns if "sleepstage" in col.lower()]
    if not stage_columns:
        raise ValueError(f"No sleepStage columns found in {csv_path}")

    manual_vectors: dict[str, np.ndarray] = {}
    for col in stage_columns:
        normalized = _normalize_stage_values(df[col].to_numpy())
        manual_vectors[col] = _collapse_stage_vector(
            normalized,
            sampling_rate_hz=sampling_rate_hz,
            time_resolution_s=time_resolution_s,
        )

    return raw_signals, manual_vectors


def load_raw_signals_from_csv(
    csv_path: Path,
) -> np.ndarray:
    if csv_path.name.startswith("._"):
        raise ValueError(f"Metadata file not supported: {csv_path}")

    df = load_signal_table(csv_path)
    required = ["EEG1", "EEG2", "EMG"]
    if not set(required).issubset(df.columns):
        raise ValueError(f"Missing required columns in {csv_path}")
    return df[required].to_numpy(dtype=float)


def _short_label_for_stem(stem: str) -> str:
    suffix_match = re.search(r"recording-\d+[_-](.+)$", stem)
    if suffix_match:
        return suffix_match.group(1)
    subses_match = re.search(r"(sub-[^_-]+[_-]ses-[^_-]+)", stem)
    if subses_match:
        return subses_match.group(1)
    return stem


def _resolve_file_labels(csv_paths: list[Path]) -> dict[Path, str]:
    proposed = {p: _short_label_for_stem(p.stem) for p in csv_paths}
    counts = Counter(proposed.values())
    return {p: (lbl if counts[lbl] == 1 else p.stem) for p, lbl in proposed.items()}


def load_manual_vectors_from_csvs(
    csv_paths: Iterable[Path],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    csv_paths = list(csv_paths)
    if not csv_paths:
        raise ValueError("No CSV paths provided")

    file_labels = _resolve_file_labels(csv_paths)
    raw_signals: np.ndarray | None = None
    manual_vectors: dict[str, np.ndarray] = {}

    for csv_path in csv_paths:
        signals, vectors = load_testing_csv(csv_path)
        if raw_signals is None:
            raw_signals = signals
        elif signals.shape != raw_signals.shape:
            raise ValueError(
                "Raw signal shapes differ across CSVs. "
                f"{csv_path.name} has {signals.shape}, expected {raw_signals.shape}."
            )

        file_label = file_labels[csv_path]
        for name, vec in vectors.items():
            label = file_label if len(vectors) == 1 else f"{file_label}:{name}"
            if label in manual_vectors:
                label = f"{csv_path.stem}:{name}"
            manual_vectors[label] = vec

    return raw_signals, manual_vectors


# --------------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------------

def align_vectors(
    somnotate_vec: np.ndarray,
    manual_vectors: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    lengths = [len(somnotate_vec)] + [len(vec) for vec in manual_vectors.values()]
    min_len = min(lengths) if lengths else 0
    if min_len == 0:
        return somnotate_vec[:0], {name: vec[:0] for name, vec in manual_vectors.items()}

    somnotate_vec = somnotate_vec[:min_len]
    manual_vectors = {name: vec[:min_len] for name, vec in manual_vectors.items()}
    return somnotate_vec, manual_vectors


def load_aligned_vectors(
    csv_dir: Path,
    predictions_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Load raw_signals, somnotate_vec, manual_vectors from disk and align lengths.

    Raises FileNotFoundError with a clear hint if either input is missing.
    """
    csv_files = list_csv_files(csv_dir)
    if not csv_files:
        raise FileNotFoundError(
            f"No CSVs in {csv_dir}. Run the data-preparation cell (ensure_csvs / ensure_somnotate_predictions)."
        )
    pred_path = find_somnotate_predictions(predictions_dir)
    if pred_path is None:
        raise FileNotFoundError(
            f"No somnotate predictions in {predictions_dir}. Run the data-preparation cell."
        )

    raw_signals, manual_vectors = load_manual_vectors_from_csvs(csv_files)
    somnotate_vec = load_somnotate_predictions(pred_path)
    somnotate_vec, manual_vectors = align_vectors(somnotate_vec, manual_vectors)
    return raw_signals, somnotate_vec, manual_vectors


# --------------------------------------------------------------------------------
# Scoring outputs: where they live, and reading them back
# --------------------------------------------------------------------------------

# Suffixes `score_recordings` writes next to each recording. Defined here so the
# naming has exactly one owner — both the writer (scoring) and the readers below
# derive paths from these rather than rebuilding the strings.
PREDICTIONS_SUFFIX = "_somnotate_predictions.parquet"
SEGMENTS_SUFFIX = "_somnotate_segments.json"
HYPNOGRAM_SUFFIX = "_somnotate_predictions.txt"
NORMALIZATION_SUFFIX = "_somnotate_normalization.npz"


def prediction_path(recording) -> Path:
    """Path to the per-epoch predictions parquet for a resolved recording."""
    return recording.output_dir / f"{recording.edf_path.stem}{PREDICTIONS_SUFFIX}"


def segments_path(recording) -> Path:
    """Path to the segment/gap metadata JSON for a resolved recording."""
    return recording.output_dir / f"{recording.edf_path.stem}{SEGMENTS_SUFFIX}"


def hypnogram_path(recording) -> Path:
    """Path to the visbrain hypnogram .txt for a resolved recording."""
    return recording.output_dir / f"{recording.edf_path.stem}{HYPNOGRAM_SUFFIX}"


def normalization_stats_path(recording) -> Path:
    """Path to the recording's own pooled normalization statistics (.npz)."""
    return recording.output_dir / f"{recording.edf_path.stem}{NORMALIZATION_SUFFIX}"


@dataclass(frozen=True)
class ScoredRef:
    """One recording that `score_recordings` has produced (or could produce) output for.

    `scored` says whether the predictions parquet actually exists yet, so a caller can
    tell "not scored" apart from "no such recording".
    """

    subject: int
    session: str
    date: str
    recording: str
    edf_path: Path
    predictions_path: Path
    segments_path: Path
    scored: bool


def _subject_number(label: str) -> int | str:
    """"sub-066" -> 66; anything without digits is returned unchanged."""
    digits = "".join(ch for ch in label if ch.isdigit())
    return int(digits) if digits else label


def find_scored(
    sub,
    date=None,
    date_range=None,
    repo_root: Path | None = None,
    scored_only: bool = True,
) -> list[ScoredRef]:
    """Find the scoring outputs for a subject/date selection.

    Accepts the same forgiving selectors as the CLI — `66`, `"066"`, `"sub-066"`,
    lists, and `"66,67"` all work, as do `date="20260707"` and
    `date_range="20260707-20260718"`.

    With `scored_only` (the default) only recordings whose predictions parquet exists
    are returned. Pass False to also see recordings that have not been scored yet —
    useful for reporting what is still outstanding.
    """
    subjects = parse_subjects(sub)
    dates = parse_dates(date) or None
    span = parse_date_range(date_range)
    if dates and span:
        raise ValueError("Use either date or date_range, not both.")

    recordings = find_recordings(repo_root, subjects, dates=dates, date_range=span)

    refs: list[ScoredRef] = []
    for rec in recordings:
        pred = prediction_path(rec)
        ref = ScoredRef(
            subject=_subject_number(rec.subject),
            session=rec.session,
            date=rec.date,
            recording=rec.edf_path.stem,
            edf_path=rec.edf_path,
            predictions_path=pred,
            segments_path=segments_path(rec),
            scored=pred.exists(),
        )
        if ref.scored or not scored_only:
            refs.append(ref)
    return refs


def _epoch_seconds(df: pd.DataFrame) -> float:
    """Duration of one epoch, from the spacing of `time_s` in a single recording.

    Falls back to somnotate's configured annotation resolution when the spacing cannot
    be measured (a one-row file, or `time_s` not loaded via `columns=`).
    """
    if "time_s" in df.columns and len(df) > 1:
        step = float(pd.Series(df["time_s"]).diff().median())
        if step > 0:
            return step
    # imported lazily: configuration pulls in matplotlib and mutates rcParams
    from ..somnotate_pipeline.utils import configuration

    return float(configuration.time_resolution)


def load_scores(
    sub,
    date=None,
    date_range=None,
    repo_root: Path | None = None,
    columns: list[str] | None = None,
    missing: str = "warn",
) -> pd.DataFrame:
    """Load per-epoch scoring results for a subject/date selection into one DataFrame.

    Reads only the predictions parquet — never the EDF — so it stays fast enough to
    call across a whole cohort.

    The result is tidy: every row is one epoch, with `subject`, `date`, `session` and
    `recording` prepended to the stored columns, so several recordings/animals can be
    concatenated and then grouped or filtered::

        df = load_scores([66, 67], date_range="20260707-20260718")
        nrem = df[df["label_output"] == 1]
        nrem.groupby(["subject", "date"]).size()

    Stored columns are `time_s`, `label`, `label_model`, `label_output`, `segment_id`,
    `kind`, and `prob_wake` / `prob_nrem` / `prob_rem` / `prob_undef`.

    Arguments:
    ----------
    columns -- read only these stored columns (the four identifier columns are always
        included). Pushed down to the parquet reader, so it saves IO as well as memory.

    missing -- what to do about selected recordings that have not been scored:
        "warn" (default), "raise", or "ignore".

    Raises FileNotFoundError if nothing in the selection has been scored, so an empty
    result is never mistaken for "no sleep found".
    """
    if missing not in ("warn", "raise", "ignore"):
        raise ValueError(f"missing must be 'warn', 'raise' or 'ignore', got {missing!r}")

    refs = find_scored(sub, date, date_range, repo_root, scored_only=False)
    if not refs:
        raise FileNotFoundError(
            f"No recordings found for subject(s) {sub!r}"
            + (f" date {date!r}" if date else "")
            + (f" range {date_range!r}" if date_range else "")
            + "."
        )

    unscored = [r for r in refs if not r.scored]
    if unscored:
        summary = ", ".join(f"sub-{r.subject:03d}/{r.date}/{r.recording}" for r in unscored[:5])
        more = "" if len(unscored) <= 5 else f" (+{len(unscored) - 5} more)"
        message = (
            f"{len(unscored)} selected recording(s) have no somnotate predictions yet: "
            f"{summary}{more}. Run `score` for them first."
        )
        if missing == "raise":
            raise FileNotFoundError(message)
        if missing == "warn":
            warnings.warn(message, UserWarning, stacklevel=2)

    read_columns = None
    if columns is not None:
        read_columns = list(dict.fromkeys(columns))

    frames: list[pd.DataFrame] = []
    for ref in refs:
        if not ref.scored:
            continue
        df = pd.read_parquet(ref.predictions_path, columns=read_columns)
        # Duration of one epoch, so callers never have to hardcode it. Derived from the
        # file's own time_s spacing rather than assumed: somnotate's annotation
        # resolution (1 s) is NOT the manual scoring epoch
        # (DEFAULT_SLEEP_STAGE_RESOLUTION_S, 10 s), and confusing the two silently
        # scales every duration. With epoch_s present, `df.epoch_s.sum()` is seconds of
        # recording whatever the resolution was.
        df.insert(0, "epoch_s", _epoch_seconds(df))
        # Identifiers first, so the frame reads left-to-right from "which recording"
        # to "what happened in it".
        df.insert(0, "recording", ref.recording)
        df.insert(0, "session", ref.session)
        df.insert(0, "date", ref.date)
        df.insert(0, "subject", ref.subject)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"Nothing in this selection has been scored yet ({len(refs)} recording(s) "
            "matched). Run `score` for them first."
        )
    return pd.concat(frames, ignore_index=True)


def load_segments(ref) -> dict:
    """Read the segment/gap metadata JSON written alongside the predictions.

    Accepts a `ScoredRef` or a path. Describes how the recording was chunked around
    dropouts; the per-epoch `segment_id` column in `load_scores` indexes into it.
    """
    path = Path(getattr(ref, "segments_path", ref))
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------------
# Scored recordings with signals (selector-driven)
# --------------------------------------------------------------------------------

def load_recording_arrays(recording, channel_labels: list[str] | None = None):
    """Load raw signals + somnotate label vector for one resolved recording.

    Returns (raw_signals, somnotate_vec). Raises FileNotFoundError if the recording
    has no somnotate predictions parquet yet.

    This reads the EDF, which is slow; for the labels alone use `load_scores`.
    """
    channel_labels = channel_labels or DEFAULT_CHANNEL_LABELS
    pred_path = prediction_path(recording)
    if not pred_path.exists():
        raise FileNotFoundError(
            f"No somnotate predictions at {pred_path}. Run score_recordings for this recording first."
        )

    # imported lazily: data_io pulls in EDF readers that are slow to import
    from ..somnotate_pipeline.io.data_io import load_raw_signals

    print(f"Loading {recording.edf_path.name}…")
    raw_signals = load_raw_signals(str(recording.edf_path), channel_labels)
    somnotate_vec = load_somnotate_vector(pred_path)
    return raw_signals, somnotate_vec


def load_scored_recording(
    subjid: int | str,
    date: int | str,
    repo_root: Path,
    recording_index: int = DEFAULT_RECORDING_INDEX,
    channel_labels: list[str] | None = None,
):
    """Resolve a scored recording by subject/date and load its raw signals + labels.

    Returns (recording, raw_signals, somnotate_vec). `recording_index` is clamped to
    the valid range rather than raising, so an out-of-range index still shows a
    recording. Raises if no recording or no predictions file exists.
    """
    repo_root = Path(repo_root)

    recordings = find_recordings(repo_root, [subjid], dates=[date])
    if not recordings:
        raise ValueError(f"No recordings found for subject {subjid} on date {date}")

    # Clamp rather than crash: an out-of-range index falls back to the nearest valid
    # recording (so recording_index=3 on a subject/date with one recording still shows it).
    clamped_index = min(max(recording_index, 0), len(recordings) - 1)
    if clamped_index != recording_index:
        print(
            f"recording_index {recording_index} out of range "
            f"({len(recordings)} recording(s) for subject {subjid} on date {date}); "
            f"showing [{clamped_index}] instead."
        )
    recording = recordings[clamped_index]
    if len(recordings) > 1:
        print(
            f"{len(recordings)} recordings for subject {subjid} on date {date}; "
            f"showing [{clamped_index}] {recording.edf_path.name}"
        )

    raw_signals, somnotate_vec = load_recording_arrays(recording, channel_labels)
    return recording, raw_signals, somnotate_vec

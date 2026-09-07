"""Path resolution and dataset discovery utilities."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

from ..config import DATA_SYMLINK, EEG_SUBDIR

SUBJECT_RE = re.compile(r"^sub-(\d{3})")
SESSION_DATE_RE = re.compile(r"^ses-([^-_]+)_date-(\d{8})$")


@dataclass(frozen=True)
class RecordingRef:
    subject: str
    session: str
    date: str
    edf_path: Path
    output_dir: Path


def normalize_subjid(subjid: int | str) -> str:
    if isinstance(subjid, int):
        return f"sub-{subjid:03d}"
    subjid = str(subjid)
    if subjid.startswith("sub-"):
        return subjid
    if subjid.isdigit():
        return f"sub-{int(subjid):03d}"
    return f"sub-{subjid}"


def _iter_subject_dirs(raw_root: Path) -> Iterable[Path]:
    if not raw_root.exists():
        return []
    return [p for p in raw_root.iterdir() if p.is_dir() and SUBJECT_RE.match(p.name)]


def _find_subject_dir(raw_root: Path, subjid: str) -> Path | None:
    for subject_dir in _iter_subject_dirs(raw_root):
        if subject_dir.name.startswith(subjid):
            return subject_dir
    return None


def _parse_session_dir(session_dir: Path) -> tuple[str | None, str | None]:
    match = SESSION_DATE_RE.match(session_dir.name)
    if not match:
        return None, None
    session, date = match.groups()
    return f"ses-{session}", date


def _date_in_filter(date: str, date_list: Iterable[str] | None, date_range: tuple[str, str] | None) -> bool:
    if date_list:
        return date in date_list
    if date_range:
        start, end = date_range
        return start <= date <= end
    return True


# Data-root resolution order (highest priority first):
#   1. HYPNOSE_EEG_RAWDATA_ROOT / HYPNOSE_EEG_DERIVATIVES_ROOT env vars (explicit override)
#   2. the shared hypnose-analysis data-location profile: server_root / hypnose_eeg / <sub>
#      (selected via that repo's scripts/set_data_location.py; honours HYPNOSE_* env too)
#   3. legacy fallback: the repo-local data/hypnose_eeg symlink under repo_root
# This lets the EEG pipeline share one machine-level data-location config with
# hypnose-analysis instead of a per-clone symlink, while still working without it.

_ENV_VAR = {"rawdata": "HYPNOSE_EEG_RAWDATA_ROOT", "derivatives": "HYPNOSE_EEG_DERIVATIVES_ROOT"}
_SHARED_ENV_VARS = (
    "HYPNOSE_SERVER_ROOT", "HYPNOSE_RAWDATA_ROOT", "HYPNOSE_DERIVATIVES_ROOT", "HYPNOSE_DATA_ROOT",
)


def _env_root(subdir: str) -> Path | None:
    val = os.getenv(_ENV_VAR[subdir])
    if not val:
        return None
    return Path(os.path.expanduser(os.path.expandvars(val))).resolve()


def _shared_eeg_root() -> Path | None:
    """EEG dataset root via hypnose-analysis's data-location system, or None.

    Only used when that system is actually configured (an active profile or a shared
    HYPNOSE_* env var); otherwise we defer to the legacy symlink so the pipeline still
    works when hypnose-analysis is absent or unconfigured.
    """
    try:
        from hypnose.io import paths as ha_paths
    except Exception:
        return None
    has_shared_env = any(os.getenv(v) for v in _SHARED_ENV_VARS)
    if not ha_paths.get_active() and not has_shared_env:
        return None
    try:
        return Path(ha_paths.get_server_root()) / EEG_SUBDIR
    except Exception:
        return None


def _resolve_data_root(subdir: str, repo_root: Path | None) -> Path:
    env = _env_root(subdir)
    if env is not None:
        return env
    shared = _shared_eeg_root()
    if shared is not None:
        return (shared / subdir).resolve()
    if repo_root is not None:
        return (Path(repo_root) / DATA_SYMLINK / subdir).resolve()
    raise FileNotFoundError(
        f"Cannot resolve EEG {subdir} root: no {_ENV_VAR[subdir]} env, no active "
        "hypnose-analysis data-location profile, and no repo_root for the symlink fallback. "
        "Run hypnose-analysis's scripts/set_data_location.py, or pass repo_root."
    )


def get_eeg_root(repo_root: Path | None = None) -> Path:
    """Root of the EEG dataset — the directory containing `rawdata/`, `derivatives/`
    and `somnotate_training/`.

    Needed because not everything lives under `derivatives/`: trained models are
    written to `derivatives/somnotate_training/<model>/`, but the labelled `.mat`
    files they are trained *from* sit at `<eeg root>/somnotate_training/`. Those are
    two different directories that happen to share a name.

    Resolution mirrors `_resolve_data_root`, env vars first:
    `HYPNOSE_EEG_ROOT` → parent of `HYPNOSE_EEG_RAWDATA_ROOT` → hypnose-analysis
    profile → the legacy `data/hypnose_eeg` symlink.
    """
    explicit = os.getenv("HYPNOSE_EEG_ROOT")
    if explicit:
        return Path(os.path.expanduser(os.path.expandvars(explicit))).resolve()

    raw_env = _env_root("rawdata")
    if raw_env is not None:
        return raw_env.parent

    shared = _shared_eeg_root()
    if shared is not None:
        return shared.resolve()

    if repo_root is not None:
        return (Path(repo_root) / DATA_SYMLINK).resolve()

    raise FileNotFoundError(
        "Cannot resolve the EEG dataset root: no HYPNOSE_EEG_ROOT or "
        "HYPNOSE_EEG_RAWDATA_ROOT env var, no active hypnose-analysis data-location "
        "profile, and no repo_root for the symlink fallback."
    )


def get_raw_root(repo_root: Path | None = None) -> Path:
    return _resolve_data_root("rawdata", repo_root)


def get_derivatives_root(repo_root: Path | None = None) -> Path:
    return _resolve_data_root("derivatives", repo_root)


def _prefer_concatenated_recording(
    edf_paths: list[Path], *, sub_label: str, session: str, date: str
) -> list[Path]:
    """When a session's ephys/ folder has multiple EDFs, score only the concatenated one.

    The glob in `find_recordings` matches every EDF that looks like a recording for
    this session -- both the original per-part files and a
    `_recording-concat.edf` produced by concatenating them (see
    `scripts/preprocessing/concatenate_recordings.py` in hypnose-eeg-analysis).
    Without this filter, a multi-part session gets scored once per original part
    *and* once for the concatenation: duplicated work, and the individual parts get
    scored with none of `prepare_recording`'s gap-aware handling between them
    (there's no real gap between two files scored independently -- they just look
    like two separate, shorter recordings).

    If more than one EDF is present but none of them is a concatenated recording,
    the session is skipped (with a warning) rather than falling back to scoring the
    individual parts -- mirrors `EdfDownsampler._prefer_concatenated_recordings` in
    hypnose-eeg-analysis's `scripts/preprocessing/downsample_recordings.py`.
    """
    if len(edf_paths) <= 1:
        return edf_paths

    concat_paths = [
        path
        for path in edf_paths
        if path.name.lower().endswith("recording-concat.edf")
        or "_recording-concat" in path.name.lower()
    ]
    if not concat_paths:
        warnings.warn(
            f"{sub_label} {session} (date {date}) has {len(edf_paths)} EDF files "
            "but none is a concatenated recording ('_recording-concat.edf'); "
            "skipping until scripts/preprocessing/concatenate_recordings.py has "
            "been run for this session.",
            UserWarning,
            stacklevel=2,
        )
        return []
    return concat_paths


def find_recordings(
    repo_root: Path,
    subjids: Iterable[int | str],
    dates: Iterable[int | str] | None = None,
    date_range: tuple[int | str, int | str] | None = None,
    output_subdir: str = "saved_results",
) -> list[RecordingRef]:
    raw_root = get_raw_root(repo_root)
    derivatives_root = get_derivatives_root(repo_root)

    normalized_dates = [str(d) for d in dates] if dates else None
    normalized_range = None
    if date_range:
        normalized_range = (str(date_range[0]), str(date_range[1]))

    results: list[RecordingRef] = []
    for subjid in subjids:
        sub_label = normalize_subjid(subjid)
        subject_dir = _find_subject_dir(raw_root, sub_label)
        if subject_dir is None:
            warnings.warn(
                f"No subject directory found for {sub_label}.",
                UserWarning,
                stacklevel=2,
            )
            continue

        found_dates: set[str] = set()
        for session_dir in subject_dir.iterdir():
            if not session_dir.is_dir():
                continue
            session, date = _parse_session_dir(session_dir)
            if not session or not date:
                continue
            if not _date_in_filter(date, normalized_dates, normalized_range):
                continue
            found_dates.add(date)

            ephys_dir = session_dir / "ephys"
            if not ephys_dir.exists():
                warnings.warn(
                    f"No ephys/ directory for {sub_label} {session} (date {date}).",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            edf_paths = sorted(ephys_dir.glob(f"{sub_label}_ses-*recording-*.edf"))
            if not edf_paths:
                pvfs_files = list(ephys_dir.glob("*.pvfs"))
                if pvfs_files:
                    warnings.warn(
                        f"No EDF file for {sub_label} {session} (date {date}); "
                        f"found only .pvfs files in {ephys_dir}. "
                        "Convert the .pvfs to .edf before scoring.",
                        UserWarning,
                        stacklevel=2,
                    )
                else:
                    warnings.warn(
                        f"No EEG data files for {sub_label} {session} (date {date}); "
                        f"{ephys_dir} contains no .edf or .pvfs files.",
                        UserWarning,
                        stacklevel=2,
                    )
                continue

            edf_paths = _prefer_concatenated_recording(
                edf_paths, sub_label=sub_label, session=session, date=date
            )
            if not edf_paths:
                continue  # warning already issued by _prefer_concatenated_recording

            for edf_path in edf_paths:
                output_dir = (
                    derivatives_root
                    / subject_dir.name
                    / session_dir.name
                    / output_subdir
                )
                results.append(
                    RecordingRef(
                        subject=sub_label,
                        session=session,
                        date=date,
                        edf_path=edf_path,
                        output_dir=output_dir,
                    )
                )

        if normalized_dates is not None:
            missing = sorted(set(normalized_dates) - found_dates)
            for missing_date in missing:
                warnings.warn(
                    f"No session directory for {sub_label} on requested date {missing_date}.",
                    UserWarning,
                    stacklevel=2,
                )

    return results

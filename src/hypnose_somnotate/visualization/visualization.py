"""Visualization utilities for Somnotate outputs."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ..somnotate._manual_state_annotation import TimeSeriesStateViewer
from ..somnotate._utils import (
    convert_state_vector_to_state_intervals,
    _get_intervals,
)
from ..somnotate_pipeline.utils import configuration
from ..somnotate_pipeline.utils.configuration import (
    plot_raw_signals,
    plot_states,
)

from ..config import (
    DEFAULT_DISTRIBUTION_BIN_WIDTH,
    DEFAULT_DISTRIBUTION_EPOCH_LENGTH_S,
    DEFAULT_EEG_CHANNEL,
    DEFAULT_RECORDING_INDEX,
    DEFAULT_SAMPLING_RATE_HZ,
    DEFAULT_VIEW_LENGTH_S,
)
from ..io.loading import load_recording_arrays, load_scored_recording
from ..io.paths import find_recordings
from ..io.style import ensure_style


def plot_recording_comparison(
    raw_signals: np.ndarray,
    sampling_rate_hz: float,
    somnotate_states: list[str],
    somnotate_intervals: list[tuple[float, float]],
    manual_states: list[str] | None = None,
    manual_intervals: list[tuple[float, float]] | None = None,
    epoch_seconds: int = 10,
):
    ensure_style()
    fig = plt.figure(figsize=(12, 7))
    from matplotlib.gridspec import GridSpec

    gs = GridSpec(5 if manual_states else 4, 1)
    data_axis = fig.add_subplot(gs[:3, 0])
    auto_axis = fig.add_subplot(gs[3, 0], sharex=data_axis)
    manual_axis = fig.add_subplot(gs[4, 0], sharex=data_axis) if manual_states else None
    fig.tight_layout(**{"rect": [0.05, 0, 1, 1], "pad": 2.0, "h_pad": 0.5})

    plot_raw_signals(
        raw_signals,
        sampling_frequency=sampling_rate_hz,
        ax=data_axis,
        linewidth=1.0,
    )

    plot_states(somnotate_states, somnotate_intervals, ax=auto_axis, linewidth=5.0)
    auto_axis.set_ylabel("Somnotate")

    if manual_states and manual_axis is not None:
        plot_states(manual_states, manual_intervals, ax=manual_axis, linewidth=5.0)
        manual_axis.set_ylabel("Manual")

    auto_axis.set_xlabel("Time [s]")
    return fig


def compare_vectors(auto_vec: np.ndarray, manual_vec: np.ndarray) -> list[tuple[int, int]]:
    auto_vec = np.abs(auto_vec)
    manual_vec = np.abs(manual_vec)
    min_len = min(len(auto_vec), len(manual_vec))
    auto_vec = auto_vec[:min_len]
    manual_vec = manual_vec[:min_len]
    is_discrepancy = auto_vec != manual_vec
    return _get_intervals(is_discrepancy)


def _states_present(state_list) -> list[str]:
    """Return configuration.state_display_order filtered to states that actually appear."""
    present = set(state_list)
    return [s for s in configuration.state_display_order if s in present]


def _shade_mismatch_intervals(
    axes: list,
    epoch_intervals: np.ndarray,
    time_resolution_s: float,
    color: str = "lightblue",
    alpha: float = 0.4,
) -> None:
    """Draw a translucent vertical band on each axis for every mismatch epoch range.

    `epoch_intervals` is an (n, 2) array of [start, stop] epoch indices (the format
    `_get_intervals` returns); converted to seconds via `time_resolution_s`.
    """
    if epoch_intervals is None or len(epoch_intervals) == 0:
        return
    intervals_s = np.asarray(epoch_intervals, dtype=float) * float(time_resolution_s)
    for ax in axes:
        for start, stop in intervals_s:
            ax.axvspan(start, stop, color=color, alpha=alpha, zorder=0, linewidth=0)


def _rolling_mean(signal: np.ndarray, sampling_rate_hz: float, window_s: float) -> np.ndarray:
    win = max(int(round(sampling_rate_hz * window_s)), 1)
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(signal.astype(float), kernel, mode="same")


def _rolling_rms(signal: np.ndarray, sampling_rate_hz: float, window_s: float) -> np.ndarray:
    return np.sqrt(_rolling_mean(signal.astype(float) ** 2, sampling_rate_hz, window_s))


def _bandpass_rms(
    signal: np.ndarray,
    sampling_rate_hz: float,
    lowcut: float,
    highcut: float,
    window_s: float,
) -> np.ndarray:
    """Banded power: bandpass, then windowed RMS — matches Spike2's `banded power`
    and is the closest time-domain analogue to somnotate's windowed STFT power.
    """
    filtered = configuration.chebychev_bandpass_filter(
        signal, lowcut=lowcut, highcut=highcut, fs=sampling_rate_hz
    )
    return _rolling_rms(filtered, sampling_rate_hz, window_s)


def _stacked_signal_plot(
    ax,
    signals: list[np.ndarray],
    sampling_rate_hz: float,
    labels: list[str],
    colors: list[str],
    linewidth: float = 0.8,
    show_range: bool = True,
    value_axis_bands: set[int] | None = None,
) -> None:
    """Plot multiple signals on one axis, stacked vertically with per-signal colors.

    First signal in the list ends up at the top of the plot (matches plot_signals convention).
    Signal names are drawn as vertical (top-to-bottom) tick labels to save width. Bands
    whose input index is in `value_axis_bands` get numeric y-axis ticks on the right
    edge (values at 10/50/90% of the band's 1st–99th-percentile scaling range); when
    `show_range` is set, every other band is annotated with the value at its top and
    bottom edge so absolute scale stays readable.
    """
    from matplotlib.transforms import blended_transform_factory

    arr = np.column_stack([s.astype(float) for s in signals])

    # robust per-signal [0,1] rescale using 1st/99th percentiles to suppress outliers
    lo = np.nanpercentile(arr, 1.0, axis=0)
    hi = np.nanpercentile(arr, 99.0, axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    arr = np.clip((arr - lo) / span, 0.0, 1.0)

    # reverse order so the first input signal is at the top
    arr = arr[:, ::-1]
    labels_rev = labels[::-1]
    colors_rev = colors[::-1]
    lo_rev = lo[::-1]
    hi_rev = hi[::-1]
    span_rev = span[::-1]

    # vertical offsets so signals don't overlap
    offsets = np.arange(arr.shape[1], dtype=float)
    arr = arr + offsets[None, :]

    n_samples = arr.shape[0]
    time = np.arange(n_samples, dtype=float) / sampling_rate_hz

    for i in range(arr.shape[1]):
        ax.plot(time, arr[:, i], color=colors_rev[i], linewidth=linewidth)
        ax.axhline(offsets[i] + 0.5, color="0.85", linewidth=0.5, linestyle=":")

    # input index -> reversed (plotted) index
    value_axis_rev = {arr.shape[1] - 1 - i for i in (value_axis_bands or ())}

    if show_range:
        # numeric scale bounds pinned to the right edge, one pair per band; the value
        # at the top of the band (hi) and at the bottom (lo). Uses the full-recording
        # percentiles the bands are normalized against.
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        for i in range(arr.shape[1]):
            if i in value_axis_rev:
                continue
            ax.text(0.998, offsets[i] + 0.96, f"{hi_rev[i]:.3g}", transform=trans,
                    ha="right", va="top", fontsize=6, color="0.45")
            ax.text(0.998, offsets[i] + 0.04, f"{lo_rev[i]:.3g}", transform=trans,
                    ha="right", va="bottom", fontsize=6, color="0.45")

    ax.set_yticks(offsets + 0.5)
    ax.set_yticklabels(labels_rev, rotation=-90, va="center")

    if value_axis_rev:
        # Real axis values for the selected bands, as right-side minor ticks. Positions
        # stay inside the band so neighbouring bands' ticks never collide.
        fractions = (0.1, 0.5, 0.9)
        tick_pos, tick_labels = [], []
        for i in sorted(value_axis_rev):
            for frac in fractions:
                tick_pos.append(offsets[i] + frac)
                tick_labels.append(f"{lo_rev[i] + frac * span_rev[i]:.3g}")
        # the 50% tick shares its position with the band's name tick; keep both
        ax.yaxis.remove_overlapping_locs = False
        ax.set_yticks(tick_pos, minor=True)
        ax.set_yticklabels(tick_labels, minor=True)
        ax.tick_params(axis="y", which="minor", left=False, labelleft=False,
                       right=True, labelright=True, labelsize=7, colors="0.3")

    ax.set_xlim(time[0], time[-1])


def _align_to_shortest(
    somnotate_vec: np.ndarray,
    manual_vectors: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    lengths = [len(somnotate_vec)] + [len(v) for v in manual_vectors.values()]
    min_len = min(lengths)
    return somnotate_vec[:min_len], {k: v[:min_len] for k, v in manual_vectors.items()}


def plot_detailed_comparison(
    raw_signals: np.ndarray,
    sampling_rate_hz: int,
    somnotate_vec: np.ndarray,
    manual_vectors: dict[str, np.ndarray] | None = None,
    scorers: list[str] | None = None,
    eeg_channel: int = DEFAULT_EEG_CHANNEL,
    view_length_s: float = DEFAULT_VIEW_LENGTH_S,
    emg_rms_window_s: float = 5.0,
    band_smoothing_window_s: float = 5.0,
    delta_band: tuple[float, float] = (0.5, 4.0),
    theta_band: tuple[float, float] = (6.0, 10.0),
    time_resolution_s: float | None = None,
):
    """Interactive state view with derived signal channels; human scorers optional.

    Signals plotted (top to bottom): EEG raw, EMG raw, EMG RMS, Delta envelope,
    Theta envelope, Theta:Delta ratio. State rows: somnotate plus whichever scorers
    are selected via `scorers` (default: all of `manual_vectors`).

    With no `manual_vectors` (the scoring case) only the somnotate row is drawn and
    no mismatch regions are shaded. When scorers are present, mismatch ROIs are
    computed against the selected ones only.

    Keyboard navigation (left/right etc.) is inherited from TimeSeriesStateViewer.
    """
    ensure_style()
    time_resolution_s = time_resolution_s or configuration.time_resolution
    manual_vectors = dict(manual_vectors or {})

    if scorers is None:
        selected = manual_vectors
    else:
        missing = [s for s in scorers if s not in manual_vectors]
        if missing:
            raise KeyError(
                f"Unknown scorer(s): {missing}. Available: {list(manual_vectors.keys())}"
            )
        selected = {s: manual_vectors[s] for s in scorers}

    if eeg_channel not in (0, 1):
        raise ValueError("eeg_channel must be 0 (EEG1) or 1 (EEG2)")

    somnotate_vec, selected = _align_to_shortest(somnotate_vec, selected)

    eeg = raw_signals[:, eeg_channel]
    emg = raw_signals[:, 2]

    print("Computing derived signal channels…")
    eeg_disp = configuration.chebychev_bandpass_filter(eeg, lowcut=0.5, highcut=30.0, fs=sampling_rate_hz)
    emg_disp = configuration.chebychev_bandpass_filter(emg, lowcut=10.0, highcut=45.0, fs=sampling_rate_hz)
    emg_rms = _rolling_rms(emg_disp, sampling_rate_hz, emg_rms_window_s)
    delta_env = _bandpass_rms(
        eeg, sampling_rate_hz, *delta_band, window_s=band_smoothing_window_s
    )
    theta_env = _bandpass_rms(
        eeg, sampling_rate_hz, *theta_band, window_s=band_smoothing_window_s
    )
    td_ratio = theta_env / (delta_env + 1e-6)

    signals = [eeg_disp, emg_disp, emg_rms, delta_env, theta_env, td_ratio]
    # Short names: the long band/window descriptions overlapped as rotated tick labels.
    labels = [
        f"EEG{eeg_channel + 1} (raw)",
        "EMG (raw)",
        "EMG RMS",
        "Delta",
        "Theta",
        "T:D Ratio",
    ]
    colors = ["green", "green", "orange", "pink", "red", "blue"]

    state_rows = 1 + len(selected)
    fig = plt.figure(figsize=(13, 8 + 1.0 * state_rows))
    from matplotlib.gridspec import GridSpec

    gs = GridSpec(1 + state_rows, 1, height_ratios=[4] + [1] * state_rows)
    data_axis = fig.add_subplot(gs[0, 0])
    auto_axis = fig.add_subplot(gs[1, 0], sharex=data_axis)
    manual_axes = [fig.add_subplot(gs[i + 2, 0], sharex=data_axis) for i in range(len(selected))]
    # Right margin leaves room for the derived channels' y-axis values.
    fig.tight_layout(**{"rect": [0.05, 0, 0.95, 1], "pad": 2.0, "h_pad": 0.6})

    _stacked_signal_plot(
        data_axis,
        signals=signals,
        sampling_rate_hz=sampling_rate_hz,
        labels=labels,
        colors=colors,
        linewidth=0.8,
        value_axis_bands={2, 3, 4, 5},
    )

    auto_states, auto_intervals = convert_state_vector_to_state_intervals(
        somnotate_vec,
        time_resolution=time_resolution_s,
        mapping=configuration.int_to_state,
    )
    # The viewer plots somnotate state intervals on auto_axis itself — don't double-plot.
    auto_axis.set_ylabel("Somnotate")

    for axis, (name, vec) in zip(manual_axes, selected.items()):
        manual_states, manual_intervals = convert_state_vector_to_state_intervals(
            vec, time_resolution=time_resolution_s, mapping=configuration.int_to_state
        )
        configuration.plot_states(manual_states, manual_intervals, ax=axis, linewidth=5.0)
        axis.set_ylabel(name)

    auto_axis.set_xlabel("Time [s]")

    regions_of_interest = None
    if selected:
        mismatch = np.zeros_like(somnotate_vec, dtype=bool)
        for vec in selected.values():
            mismatch |= (somnotate_vec != vec)
        regions_of_interest = _get_intervals(mismatch)
        _shade_mismatch_intervals(
            [auto_axis, *manual_axes], regions_of_interest, time_resolution_s
        )

    viewer = TimeSeriesStateViewer(
        data_axis,
        auto_axis,
        interval_to_state=zip(auto_intervals, auto_states),
        regions_of_interest=regions_of_interest,
        state_to_color=configuration.state_to_color,
        state_display_order=_states_present(auto_states),
        default_selection_length=configuration.default_selection_length,
        default_view_length=view_length_s,
    )
    return fig, viewer


def plot_scoring_detailed(
    subjid: int | str,
    date: int | str,
    repo_root: Path,
    manual_vectors: dict[str, np.ndarray] | None = None,
    recording_index: int = DEFAULT_RECORDING_INDEX,
    channel_labels: list[str] | None = None,
    sampling_rate_hz: int = DEFAULT_SAMPLING_RATE_HZ,
    **plot_kwargs,
):
    """Load one scored recording by subject/date and show the detailed state view.

    Finds the recording, reads its EDF and the somnotate predictions written by
    `score_recordings`, and hands both to `plot_detailed_comparison`. Human scorer
    vectors are optional — pass `manual_vectors` to add scorer rows and mismatch
    shading, or omit it to view the somnotate scoring on its own.

    `recording_index` selects among multiple EDFs for the same subject/date.
    Extra keyword arguments (eeg_channel, view_length_s, delta_band, …) pass
    through to `plot_detailed_comparison`.
    """
    recording, raw_signals, somnotate_vec = load_scored_recording(
        subjid, date, repo_root,
        recording_index=recording_index,
        channel_labels=channel_labels,
    )
    return plot_detailed_comparison(
        raw_signals,
        sampling_rate_hz=sampling_rate_hz,
        somnotate_vec=somnotate_vec,
        manual_vectors=manual_vectors,
        **plot_kwargs,
    )


# Model-int → (display name, colour) for the three scored vigilance states.
# somnotate labels are stored as absolute model ints (1=awake, 2=non-REM, 3=REM,
# 0=undefined); undefined/gap epochs are dropped from the distributions.
_STATE_CODE_TO_NAME = {1: "Wake", 2: "NREM", 3: "REM"}
_STATE_NAME_TO_COLOR = {"Wake": "red", "NREM": "blue", "REM": "gold"}


def _per_epoch_band_metrics(
    band_signal: np.ndarray,
    samples_per_epoch: int,
    reducer: str,
) -> np.ndarray:
    """Split a 1-D signal into non-overlapping epochs and reduce each to one scalar.

    reducer="power" -> mean square (band power); "rms" -> sqrt of mean square.
    A trailing partial epoch (shorter than samples_per_epoch) is dropped.
    """
    n_epochs = len(band_signal) // samples_per_epoch
    if n_epochs == 0:
        return np.empty(0, dtype=float)
    trimmed = band_signal[: n_epochs * samples_per_epoch].astype(float)
    epochs = trimmed.reshape(n_epochs, samples_per_epoch)
    mean_square = np.mean(epochs ** 2, axis=1)
    return mean_square if reducer == "power" else np.sqrt(mean_square)


_METRIC_KEYS = ["delta", "theta", "td", "emg"]


def _as_list(value) -> list | None:
    """Normalise a scalar-or-iterable argument to a list (or None)."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _display_inline(fig) -> None:
    """Render a standalone Figure into the notebook output as a PNG.

    Uses IPython display so it works under any active matplotlib backend (including
    `%matplotlib qt`, which otherwise routes figures to external windows). No-op when
    not running inside IPython.
    """
    try:
        import io
        from IPython.display import Image, display
    except ImportError:
        return
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    display(Image(data=buf.read()))


def _compute_state_epoch_metrics(
    raw_signals: np.ndarray,
    somnotate_vec: np.ndarray,
    sampling_rate_hz: int,
    epoch_length_s: float,
    eeg_channel: int,
    delta_band: tuple[float, float],
    theta_band: tuple[float, float],
    emg_band: tuple[float, float],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[int, int]]:
    """Per-state, per-epoch metric arrays for one recording, plus label-count occupancy.

    Only Wake/NREM/REM samples enter the metrics; undefined (0) epochs are dropped.
    Returns (per_state, occupancy_counts) where occupancy_counts maps state code
    (0/1/2/3) to number of scored labels of that state. Identical per-recording maths to
    the original single-recording implementation.
    """
    fs = sampling_rate_hz
    time_resolution_s = configuration.time_resolution
    eeg = raw_signals[:, eeg_channel]
    emg = raw_signals[:, 2]

    # Filter the continuous recording BEFORE splitting by state, so state stitching
    # never introduces filter edge artefacts at the concatenation boundaries.
    delta_sig = configuration.chebychev_bandpass_filter(eeg, lowcut=delta_band[0], highcut=delta_band[1], fs=fs)
    theta_sig = configuration.chebychev_bandpass_filter(eeg, lowcut=theta_band[0], highcut=theta_band[1], fs=fs)
    emg_sig = configuration.chebychev_bandpass_filter(emg, lowcut=emg_band[0], highcut=emg_band[1], fs=fs)

    # Expand the per-epoch state labels to a per-sample vector, then align lengths.
    state_abs = np.abs(somnotate_vec).astype(int)
    samples_per_label = int(round(time_resolution_s * fs))
    sample_state = np.repeat(state_abs, samples_per_label)
    n = min(len(sample_state), len(delta_sig))
    sample_state = sample_state[:n]
    delta_sig, theta_sig, emg_sig = delta_sig[:n], theta_sig[:n], emg_sig[:n]

    samples_per_epoch = int(round(epoch_length_s * fs))
    if samples_per_epoch < 1:
        raise ValueError("epoch_length_s * sampling_rate_hz must be >= 1 sample")

    eps = 1e-12
    per_state: dict[str, dict[str, np.ndarray]] = {}
    for code, name in _STATE_CODE_TO_NAME.items():
        mask = sample_state == code
        delta_power = _per_epoch_band_metrics(delta_sig[mask], samples_per_epoch, "power")
        theta_power = _per_epoch_band_metrics(theta_sig[mask], samples_per_epoch, "power")
        emg_rms = _per_epoch_band_metrics(emg_sig[mask], samples_per_epoch, "rms")
        td_ratio = theta_power / (delta_power + eps)
        per_state[name] = {
            "delta": delta_power,
            "theta": theta_power,
            "td": td_ratio,
            "emg": emg_rms,
        }

    occupancy_counts = {code: int(np.sum(state_abs == code)) for code in (0, 1, 2, 3)}
    return per_state, occupancy_counts


def _render_state_distribution_figure(
    per_state: dict[str, dict[str, np.ndarray]],
    title: str,
    epoch_length_s: float,
    bin_width: float,
    normalize_percentiles: tuple[float, float],
    delta_band: tuple[float, float],
    theta_band: tuple[float, float],
    emg_band: tuple[float, float],
):
    """Build a standalone 2x2 histogram figure from aggregated per-state metrics.

    Metrics are normalised to [0, 1] using percentiles pooled across all states, then
    histogrammed with `bin_width` bins on a shared 0–1 axis (Wake=red, NREM=blue,
    REM=gold). Returns (fig, norm_bounds); fig is a bare matplotlib Figure (not tied to
    pyplot / any interactive backend) so it can be rendered inline without a Qt window.
    """
    from matplotlib.figure import Figure

    lo_pct, hi_pct = normalize_percentiles
    norm_bounds: dict[str, tuple[float, float]] = {}
    for key in _METRIC_KEYS:
        pooled = np.concatenate([per_state[name][key] for name in _STATE_CODE_TO_NAME.values()])
        if pooled.size == 0:
            norm_bounds[key] = (0.0, 1.0)
            continue
        lo = float(np.percentile(pooled, lo_pct))
        hi = float(np.percentile(pooled, hi_pct))
        if hi <= lo:
            hi = lo + 1.0
        norm_bounds[key] = (lo, hi)

    def _normalize(values: np.ndarray, key: str) -> np.ndarray:
        lo, hi = norm_bounds[key]
        return np.clip((values - lo) / (hi - lo), 0.0, 1.0)

    metric_titles = {
        "delta": f"Delta power ({delta_band[0]:g}-{delta_band[1]:g} Hz)",
        "theta": f"Theta power ({theta_band[0]:g}-{theta_band[1]:g} Hz)",
        "td": "Theta : Delta ratio",
        "emg": f"EMG RMS ({emg_band[0]:g}-{emg_band[1]:g} Hz)",
    }
    metric_xlabel = {
        "delta": "Normalised power",
        "theta": "Normalised power",
        "td": "Normalised ratio",
        "emg": "Normalised amplitude",
    }

    bins = np.arange(0.0, 1.0 + bin_width, bin_width)
    fig = Figure(figsize=(12, 8))
    axes = fig.subplots(2, 2).ravel()

    for ax, key in zip(axes, _METRIC_KEYS):
        lo, hi = norm_bounds[key]
        for name in _STATE_CODE_TO_NAME.values():
            values = per_state[name][key]
            if values.size == 0:
                continue
            normed = _normalize(values, key)
            weights = np.full(values.shape, 100.0 / values.size)
            ax.hist(normed, bins=bins, weights=weights, histtype="step",
                    linewidth=1.8, color=_STATE_NAME_TO_COLOR[name], label=name)
            ax.hist(normed, bins=bins, weights=weights, histtype="stepfilled",
                    alpha=0.15, color=_STATE_NAME_TO_COLOR[name])
        ax.set_title(metric_titles[key])
        ax.set_xlabel(f"{metric_xlabel[key]}  [{lo:.3g} … {hi:.3g}]")
        ax.set_ylabel("% of epochs")
        ax.set_xlim(0.0, 1.0)
        ax.legend(frameon=False, fontsize=9)

    fig.suptitle(f"{title} — state metric distributions ({epoch_length_s:g}s epochs)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig, norm_bounds


# Short per-metric labels for the raw per-state grid (rows = state, cols = metric).
_METRIC_SHORT = {
    "delta": "Delta power",
    "theta": "Theta power",
    "td": "Theta:Delta",
    "emg": "EMG RMS",
}


def _render_state_raw_distribution_figure(
    per_state: dict[str, dict[str, np.ndarray]],
    title: str,
    epoch_length_s: float,
    bin_width: float,
    range_percentiles: tuple[float, float],
):
    """Per-state, raw-unit histograms — one row per state, one column per metric.

    Unlike the normalised overlay, each panel keeps the metric's *raw* values. Every
    column (metric) shares one X range — that metric's `range_percentiles` (1st–99th by
    default) pooled across all three states — so the states line up on a common axis and
    are directly comparable; Y is % of that state's epochs. `bin_width` is reused as a
    fraction of that p1–p99 range, so it yields the same bin count as the overlay
    (0.05 → 20 bins). A dashed line marks each state's mean; μ/σ are printed in each title
    so location and spread are readable directly. Epochs in the outer tails fall outside
    the plotted range (so bars sum to slightly under 100%). Returns a bare Figure (no
    pyplot / interactive backend).
    """
    from matplotlib.figure import Figure

    states = list(_STATE_CODE_TO_NAME.values())
    lo_pct, hi_pct = range_percentiles
    n_bins = max(1, round(1.0 / bin_width))

    # Shared X range per metric: p1–p99 of the values pooled across all states, so the
    # three rows use identical axes/bins and can be compared column-wise.
    metric_range: dict[str, tuple[float, float]] = {}
    for key in _METRIC_KEYS:
        pooled = np.concatenate([per_state[s][key] for s in states]) if states else np.empty(0)
        if pooled.size == 0:
            metric_range[key] = (0.0, 1.0)
            continue
        lo = float(np.percentile(pooled, lo_pct))
        hi = float(np.percentile(pooled, hi_pct))
        if hi <= lo:
            hi = lo + 1e-9
        metric_range[key] = (lo, hi)

    fig = Figure(figsize=(4 * len(_METRIC_KEYS), 3 * len(states)))
    axes = fig.subplots(len(states), len(_METRIC_KEYS), squeeze=False)

    for r, state in enumerate(states):
        color = _STATE_NAME_TO_COLOR[state]
        for c, key in enumerate(_METRIC_KEYS):
            ax = axes[r][c]
            values = per_state[state][key]
            short = _METRIC_SHORT[key]
            lo, hi = metric_range[key]
            bins = np.linspace(lo, hi, n_bins + 1)
            ax.set_xlim(lo, hi)
            ax.set_xlabel(short)
            ax.set_ylabel(f"{state}\n% of epochs" if c == 0 else "% of epochs")
            if values.size == 0:
                ax.set_title(f"{state} · {short}\n(no epochs)", fontsize=9)
                continue

            weights = np.full(values.shape, 100.0 / values.size)
            ax.hist(values, bins=bins, weights=weights, color=color,
                    alpha=0.75, edgecolor="white", linewidth=0.3)

            mean = float(np.mean(values))
            std = float(np.std(values))
            ax.axvline(mean, color="black", linestyle="--", linewidth=1.0)
            ax.set_title(f"{state} · {short}\nμ={mean:.3g}  σ={std:.3g}", fontsize=9)

    fig.suptitle(f"{title} — per-state raw distributions ({epoch_length_s:g}s epochs)")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def plot_state_distributions(
    subjid: int | str | list,
    date: int | str | list | None,
    repo_root: Path,
    epoch_length_s: float = DEFAULT_DISTRIBUTION_EPOCH_LENGTH_S,
    bin_width: float = DEFAULT_DISTRIBUTION_BIN_WIDTH,
    show_raw_distributions: bool = True,
    eeg_channel: int = DEFAULT_EEG_CHANNEL,
    delta_band: tuple[float, float] = (0.5, 4.0),
    theta_band: tuple[float, float] = (6.0, 10.0),
    emg_band: tuple[float, float] = (10.0, 45.0),
    normalize_percentiles: tuple[float, float] = (1.0, 99.0),
    channel_labels: list[str] | None = None,
    sampling_rate_hz: int = DEFAULT_SAMPLING_RATE_HZ,
    inline: bool = True,
    save: bool = False,
    save_name: str = "state_distributions",
    save_dpi: int = 600,
):
    """Per-state distributions of delta power, theta power, T:D ratio and EMG RMS.

    `subjid` and `date` accept a single value or a list. One figure is produced per
    (subject, date) session — a session's multiple EDF recordings are pooled into that
    session's distribution — so e.g. 2 subjects × 3 dates yields up to 6 figures.

    For each session: band-filter the continuous EEG/EMG, group samples by vigilance
    state (Wake/NREM/REM; undefined/gap epochs are dropped entirely), stitch each
    state's samples into a continuous stream, epoch it into `epoch_length_s` windows,
    and reduce every epoch to delta power, theta power, theta:delta ratio, and EMG RMS.

    Two figures are produced per session:
      * the normalised overlay — each metric normalised to [0, 1] using
        `normalize_percentiles` pooled across states (not per state, so between-state
        differences are preserved) and histogrammed with `bin_width` bins on a shared
        0–1 axis; the absolute value range mapping to 0/1 is shown in each panel's label.
      * (when `show_raw_distributions=True`) a per-state raw grid — one row per state,
        one column per metric, showing each metric's raw values over that state's own
        1st–99th percentile range with a dashed mean line and μ/σ in the title, so the
        real location and spread per state are readable. `bin_width` is reused there as a
        fraction of the p1–p99 range (0.05 → 20 bins).

    Occupancy printed per session is the % of scored Wake/NREM/REM epochs (undefined
    excluded, so the three sum to 100%).

    With `inline=True` (default) each figure is rendered into the notebook output as a
    PNG regardless of the active matplotlib backend (so it works alongside
    `%matplotlib qt`). Set `inline=False` to skip display and just return the figures.

    With `save=True` both figures are written as PDF (named `{save_name}_norm_…` /
    `{save_name}_raw_…`, at `save_dpi`) into the session's `figures/` directory —
    `derivatives/sub-*/ses-*/figures`, i.e. next to `saved_results`. Figure styling
    follows the active matplotlib rcParams, so call hypnose-behavior-analysis's `use_style()` at
    the top of the notebook to apply the house style.

    Returns a list of dicts, one per session, each with keys: `subject`, `date`,
    `recordings` (EDF names pooled), `fig` (normalised overlay), `raw_fig` (per-state
    raw grid, or None), `per_state`, `norm_bounds`, `state_occupancy` (% of scored
    Wake/NREM/REM epochs), `figures_dir` (the session figures dir), and `saved_paths`
    (PDFs written, empty unless `save=True`).
    """
    if eeg_channel not in (0, 1):
        raise ValueError("eeg_channel must be 0 (EEG1) or 1 (EEG2)")

    ensure_style()

    repo_root = Path(repo_root)
    subjids = _as_list(subjid)
    dates = _as_list(date)

    recordings = find_recordings(repo_root, subjids, dates=dates)
    if not recordings:
        raise ValueError(f"No recordings found for subject(s) {subjids} date(s) {dates}")

    # Group recordings by (subject, date), preserving discovery order.
    groups: dict[tuple[str, str], list] = {}
    for rec in recordings:
        groups.setdefault((rec.subject, rec.date), []).append(rec)

    time_resolution_s = configuration.time_resolution
    outputs = []
    for (subject, date_str), recs in groups.items():
        # Pool per-epoch metrics and occupancy across the session's recordings.
        pooled_state: dict[str, dict[str, list]] = {
            name: {key: [] for key in _METRIC_KEYS} for name in _STATE_CODE_TO_NAME.values()
        }
        occupancy_counts = {code: 0 for code in (0, 1, 2, 3)}
        used_recordings: list[str] = []

        for rec in recs:
            try:
                raw_signals, somnotate_vec = load_recording_arrays(rec, channel_labels)
            except FileNotFoundError as exc:
                warnings.warn(str(exc), UserWarning, stacklevel=2)
                continue
            print(f"Computing metrics for {rec.edf_path.name}…")
            per_state, occ = _compute_state_epoch_metrics(
                raw_signals, somnotate_vec, sampling_rate_hz, epoch_length_s,
                eeg_channel, delta_band, theta_band, emg_band,
            )
            for name in _STATE_CODE_TO_NAME.values():
                for key in _METRIC_KEYS:
                    pooled_state[name][key].append(per_state[name][key])
            for code in occupancy_counts:
                occupancy_counts[code] += occ[code]
            used_recordings.append(rec.edf_path.name)

        if not used_recordings:
            warnings.warn(
                f"No scored recordings for {subject} date-{date_str}; skipping.",
                UserWarning, stacklevel=2,
            )
            continue

        per_state = {
            name: {key: np.concatenate(pooled_state[name][key]) for key in _METRIC_KEYS}
            for name in _STATE_CODE_TO_NAME.values()
        }

        # ----- state occupancy printout (scored Wake/NREM/REM only; undefined excluded) -----
        scored_total = occupancy_counts[1] + occupancy_counts[2] + occupancy_counts[3]
        rec_note = used_recordings[0] if len(used_recordings) == 1 else f"{len(used_recordings)} recordings"
        print(f"\n[{subject} date-{date_str}] {rec_note}")
        print(f"State occupancy (of {scored_total} scored Wake/NREM/REM {time_resolution_s:g}s epochs):")
        for code, name in _STATE_CODE_TO_NAME.items():
            pct = 100.0 * occupancy_counts[code] / scored_total if scored_total else 0.0
            n_ep = len(per_state[name]["delta"])
            print(f"  {name:<5s}: {pct:5.1f}%   ({n_ep} analysis epochs of {epoch_length_s:g}s)")

        fig, norm_bounds = _render_state_distribution_figure(
            per_state,
            title=f"{subject} date-{date_str}",
            epoch_length_s=epoch_length_s,
            bin_width=bin_width,
            normalize_percentiles=normalize_percentiles,
            delta_band=delta_band,
            theta_band=theta_band,
            emg_band=emg_band,
        )
        if inline:
            _display_inline(fig)

        raw_fig = None
        if show_raw_distributions:
            raw_fig = _render_state_raw_distribution_figure(
                per_state,
                title=f"{subject} date-{date_str}",
                epoch_length_s=epoch_length_s,
                bin_width=bin_width,
                range_percentiles=normalize_percentiles,
            )
            if inline:
                _display_inline(raw_fig)

        # Figures dir sits next to saved_results: derivatives/sub-*/ses-*/figures.
        # recs[0].output_dir is .../ses-*/saved_results, so its parent is the session dir.
        figures_dir = recs[0].output_dir.parent / "figures"

        saved_paths: list[Path] = []
        if save:
            # Routed through hypnose-behavior-analysis's save_figure so these PDFs match the
            # figures every other Hypnose repo emits (style, dpi, fonttype). The
            # session figures dir is passed explicitly: it is already resolved here
            # from the recording, and it is more specific than a subject/date lookup.
            from ..io.save import save_figure

            # save_figure tags filenames from numeric subject ids (`int(s):03d`),
            # whereas RecordingRef.subject is the label "sub-001" — pass the number
            # so the tag round-trips to the same "sub-001" the old code wrote.
            sub_digits = "".join(ch for ch in subject if ch.isdigit())
            subj_tag_value = int(sub_digits) if sub_digits else subject

            to_save = [("norm", fig)] + ([("raw", raw_fig)] if raw_fig is not None else [])
            for suffix, f in to_save:
                saved_paths.append(
                    save_figure(
                        f,
                        f"{save_name}_{suffix}",
                        subjids=subj_tag_value,
                        dates=date_str,
                        fig_dir=figures_dir,
                        dpi=save_dpi,
                    )
                )
            print(f"Saved {len(saved_paths)} figure(s) to {figures_dir}")

        occupancy_pct = {
            name: (100.0 * occupancy_counts[code] / scored_total if scored_total else 0.0)
            for code, name in _STATE_CODE_TO_NAME.items()
        }
        outputs.append({
            "subject": subject,
            "date": date_str,
            "recordings": used_recordings,
            "fig": fig,
            "raw_fig": raw_fig,
            "per_state": per_state,
            "norm_bounds": norm_bounds,
            "state_occupancy": occupancy_pct,
            "figures_dir": figures_dir,
            "saved_paths": saved_paths,
        })

    return outputs

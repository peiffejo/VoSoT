import csv
import datetime
from collections import Counter
from dataclasses import dataclass

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann

from database import DEFAULT_DB_PATH, get_session_samples, get_session_stats


@dataclass
class RecordingData:
    timestamps: list
    volumes: list
    barks: list
    sample_rate: float


def load_recording_data(log_path="bark_log.csv"):
    timestamps = []
    volumes = []
    barks = []

    with open(log_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            timestamps.append(datetime.datetime.fromtimestamp(float(row["timestamp"])))
            volumes.append(float(row["volume"]))
            barks.append(int(row["is_bark"]))

    sample_rate = 16000
    if len(timestamps) > 1:
        total_time = (timestamps[-1] - timestamps[0]).total_seconds()
        if total_time > 0:
            sample_rate = len(timestamps) / total_time

    return RecordingData(timestamps=timestamps, volumes=volumes, barks=barks, sample_rate=sample_rate)


def load_session_data(session_id: int):
    samples = get_session_samples(session_id)
    if not samples:
        return RecordingData(timestamps=[], volumes=[], barks=[], sample_rate=16000)

    timestamps = [datetime.datetime.fromtimestamp(s.timestamp) for s in samples]
    volumes = [s.volume for s in samples]
    barks = [s.is_bark for s in samples]

    sample_rate = 16000
    if len(timestamps) > 1:
        total_time = timestamps[-1] - timestamps[0]
        total_seconds = total_time.total_seconds()
        if total_seconds > 0:
            sample_rate = len(timestamps) / total_seconds

    return RecordingData(timestamps=timestamps, volumes=volumes, barks=barks, sample_rate=sample_rate)


def summarize_recording(data):
    if not data.timestamps:
        return "No recording data available yet."

    duration = data.timestamps[-1] - data.timestamps[0]
    bark_count = sum(data.barks)
    average_volume = sum(data.volumes) / len(data.volumes)
    peak_volume = max(data.volumes)
    return (
        f"Samples: {len(data.volumes):,} | Duration: {duration} | "
        f"Avg volume: {average_volume:.1f} | Peak volume: {peak_volume:.1f} | "
        f"Bark events: {bark_count}"
    )


def _add_spectrogram(data, ax):
    if len(data.volumes) < 128:
        return

    signal = np.array(data.volumes, dtype=np.float64)
    signal -= signal.mean()
    if signal.std() > 0:
        signal /= signal.std()

    fs = max(data.sample_rate, 1)
    nperseg = min(128, len(signal) // 4)
    if nperseg < 8:
        return

    window = hann(nperseg)
    hop = nperseg // 2
    SFT = ShortTimeFFT(window, hop, fs, scale_to="psd")
    Z = SFT.stft(signal)
    t = SFT.t(len(signal))
    f = SFT.f

    tdt = [data.timestamps[0] + datetime.timedelta(seconds=float(x)) for x in t]

    extent = [mdates.date2num(tdt[0]), mdates.date2num(tdt[-1]), f[0], f[-1]]
    ax.imshow(
        np.abs(Z),
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="magma",
    )
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Volume Spectrogram")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))


def build_plot(data, threshold):
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[3, 1])

    volume_ax = fig.add_subplot(gs[0, 0])
    bark_ax = fig.add_subplot(gs[1, 0], sharex=volume_ax)
    spectrogram_ax = fig.add_subplot(gs[2, 0], sharex=volume_ax)
    hist_ax = fig.add_subplot(gs[0, 1], sharey=volume_ax)
    cumul_ax = fig.add_subplot(gs[1, 1])

    volume_ax.plot(
        data.timestamps,
        data.volumes,
        color="#3498db",
        linewidth=1.2,
        label="Volume",
    )
    volume_ax.axhline(
        y=threshold,
        color="#e74c3c",
        linestyle="--",
        linewidth=1.5,
        label="Threshold",
    )

    bark_timestamps = [
        timestamp for timestamp, bark in zip(data.timestamps, data.barks) if bark
    ]
    bark_volumes = [
        volume for volume, bark in zip(data.volumes, data.barks) if bark
    ]
    if bark_timestamps:
        volume_ax.scatter(
            bark_timestamps,
            bark_volumes,
            color="#f39c12",
            s=30,
            zorder=5,
            label="Bark detected",
        )

    volume_ax.fill_between(
        data.timestamps,
        0,
        data.volumes,
        where=[bark == 1 for bark in data.barks],
        color="#f1c40f",
        alpha=0.35,
    )
    volume_ax.set_ylabel("Volume (RMS)")
    volume_ax.set_title("Volume Timeline")
    volume_ax.legend(loc="upper right", framealpha=0.9)
    volume_ax.grid(alpha=0.2)

    bark_by_minute = Counter(
        timestamp.replace(second=0, microsecond=0)
        for timestamp, bark in zip(data.timestamps, data.barks)
        if bark
    )
    if bark_by_minute:
        minute_bins = sorted(bark_by_minute)
        counts = [bark_by_minute[m] for m in minute_bins]
        bars = bark_ax.bar(minute_bins, counts, width=0.0005, color="#2ecc71", alpha=0.8)
        for bar, count in zip(bars, counts):
            if count > 0:
                bark_ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    str(count),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    bark_ax.set_ylabel("Barks / min")
    bark_ax.set_title("Bark Events Per Minute")
    bark_ax.grid(alpha=0.2)

    _add_spectrogram(data, spectrogram_ax)
    spectrogram_ax.set_xlabel("Time")

    hist_ax.hist(
        data.volumes,
        bins=40,
        orientation="horizontal",
        color="#9b59b6",
        alpha=0.7,
        edgecolor="white",
    )
    hist_ax.axhline(y=threshold, color="#e74c3c", linestyle="--", linewidth=1.5)
    hist_ax.set_title("Volume Distribution")
    hist_ax.grid(alpha=0.2)
    hist_ax.tick_params(labelleft=False)

    if data.barks:
        cumulative = np.cumsum(data.barks)
        cumul_ax.plot(
            data.timestamps,
            cumulative,
            color="#e67e22",
            linewidth=2,
        )
        cumul_ax.fill_between(
            data.timestamps,
            0,
            cumulative,
            color="#e67e22",
            alpha=0.2,
        )
        cumul_ax.set_ylabel("Total Barks")
        cumul_ax.set_title("Cumulative Bark Events")
        cumul_ax.grid(alpha=0.2)
    else:
        cumul_ax.set_visible(False)

    for ax in (volume_ax, bark_ax, spectrogram_ax):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    fig.autofmt_xdate()
    return fig


def create_plot(log_path="bark_log.csv", threshold=1000):
    data = load_recording_data(log_path)
    if not data.timestamps:
        raise ValueError("No recording data available yet.")
    figure = build_plot(data, threshold)
    return figure, summarize_recording(data)


def create_plot_for_session(session_id: int, threshold=1000):
    data = load_session_data(session_id)
    if not data.timestamps:
        raise ValueError("No recording data available yet.")
    figure = build_plot(data, threshold)
    return figure, summarize_recording(data)


def create_comparison_plot(session_ids: list[int], threshold=1000):
    if len(session_ids) < 2:
        raise ValueError("Need at least 2 sessions to compare.")

    all_data = []
    for sid in session_ids:
        data = load_session_data(sid)
        if data.timestamps:
            all_data.append((sid, data))

    if len(all_data) < 2:
        raise ValueError("Not enough sessions with data to compare.")

    n = len(all_data)
    fig = plt.figure(figsize=(16, 4 + n * 3.5), constrained_layout=True)
    gs = fig.add_gridspec(n + 2, 1)

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

    for idx, (sid, data) in enumerate(all_data):
        ax = fig.add_subplot(gs[idx, 0])
        color = colors[idx % len(colors)]

        relative_times = [(t - data.timestamps[0]).total_seconds() / 60 for t in data.timestamps]
        ax.plot(relative_times, data.volumes, color=color, linewidth=1.0, label=f"Session {sid}")
        ax.axhline(y=threshold, color="#555555", linestyle="--", linewidth=1.0, alpha=0.7)

        bark_times = [
            (t - data.timestamps[0]).total_seconds() / 60
            for t, b in zip(data.timestamps, data.barks) if b
        ]
        bark_vols = [v for v, b in zip(data.volumes, data.barks) if b]
        if bark_times:
            ax.scatter(bark_times, bark_vols, color=color, s=25, zorder=5, marker="x")

        duration = (data.timestamps[-1] - data.timestamps[0]).total_seconds() / 60
        avg_vol = sum(data.volumes) / len(data.volumes)
        peak_vol = max(data.volumes)
        ax.set_title(
            f"Session {sid}: {duration:.1f} min | Avg vol {avg_vol:.1f} | Peak {peak_vol:.1f} | Barks {sum(data.barks)}",
            fontsize=10,
            loc="left",
        )
        ax.set_ylabel("Volume (RMS)")
        ax.grid(alpha=0.2)
        ax.set_xlim(left=0)

    ax_barks = fig.add_subplot(gs[n, 0])
    labels = [f"Session {sid}" for sid, _ in all_data]
    bark_counts = [sum(data.barks) for _, data in all_data]
    bars = ax_barks.bar(labels, bark_counts, color=colors[:n], alpha=0.8, edgecolor="white")
    for bar, count in zip(bars, bark_counts):
        ax_barks.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha="center", va="bottom")
    ax_barks.set_ylabel("Total Barks")
    ax_barks.set_title("Total Bark Events per Session")
    ax_barks.grid(alpha=0.2)

    ax_stats = fig.add_subplot(gs[n + 1, 0])
    x = np.arange(n)
    width = 0.35
    avg_volumes = [sum(data.volumes) / len(data.volumes) for _, data in all_data]
    peak_volumes = [max(data.volumes) for _, data in all_data]
    ax_stats.bar(x - width / 2, avg_volumes, width, label="Avg volume", color="#3498db", alpha=0.8)
    ax_stats.bar(x + width / 2, peak_volumes, width, label="Peak volume", color="#e74c3c", alpha=0.8)
    ax_stats.set_xticks(x)
    ax_stats.set_xticklabels(labels)
    ax_stats.set_ylabel("Volume (RMS)")
    ax_stats.set_title("Average vs Peak Volume per Session")
    ax_stats.legend()
    ax_stats.grid(alpha=0.2)

    fig.suptitle(f"Session Comparison (threshold = {threshold})", fontsize=14, fontweight="bold")

    total_barks = sum(sum(data.barks) for _, data in all_data)
    total_duration = sum(
        (data.timestamps[-1] - data.timestamps[0]).total_seconds() / 60
        for _, data in all_data
    )
    summary = (
        f"Comparing {n} sessions | Total barks: {total_barks} | "
        f"Combined duration: {total_duration:.1f} min"
    )
    return fig, summary


def build_plotly(data, threshold):
    """Create an interactive Plotly figure for a single session."""
    if not data.timestamps:
        raise ValueError("No recording data available yet.")

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Volume Timeline",
            "Volume Distribution",
            "Bark Events Per Minute",
            "Cumulative Bark Events",
            "Volume Spectrogram",
            ""
        ),
        specs=[
            [{"type": "scatter"}, {"type": "histogram"}],
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "heatmap", "colspan": 2}, None]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Volume Timeline (row 1, col 1)
    fig.add_trace(
        go.Scatter(
            x=data.timestamps,
            y=data.volumes,
            mode="lines",
            name="Volume",
            line=dict(color="#3498db", width=1.2),
            hovertemplate="Time: %{x}<br>Volume: %{y:.1f}<extra></extra>"
        ),
        row=1, col=1
    )
    fig.add_hline(y=threshold, line_dash="dash", line_color="#e74c3c", row=1, col=1)

    bark_timestamps = [t for t, b in zip(data.timestamps, data.barks) if b]
    bark_volumes = [v for v, b in zip(data.volumes, data.barks) if b]
    if bark_timestamps:
        fig.add_trace(
            go.Scatter(
                x=bark_timestamps,
                y=bark_volumes,
                mode="markers",
                name="Bark detected",
                marker=dict(color="#f39c12", size=8, symbol="x"),
                hovertemplate="Time: %{x}<br>Volume: %{y:.1f}<extra></extra>"
            ),
            row=1, col=1
        )

    # Volume Distribution (row 1, col 2)
    fig.add_trace(
        go.Histogram(
            y=data.volumes,
            nbinsy=40,
            marker_color="#9b59b6",
            opacity=0.7,
            name="Volume dist",
            showlegend=False
        ),
        row=1, col=2
    )
    fig.add_hline(y=threshold, line_dash="dash", line_color="#e74c3c", row=1, col=2)

    # Bark Events Per Minute (row 2, col 1)
    bark_by_minute = Counter(
        timestamp.replace(second=0, microsecond=0)
        for timestamp, bark in zip(data.timestamps, data.barks)
        if bark
    )
    if bark_by_minute:
        minute_bins = sorted(bark_by_minute)
        counts = [bark_by_minute[m] for m in minute_bins]
        fig.add_trace(
            go.Bar(
                x=minute_bins,
                y=counts,
                marker_color="#2ecc71",
                opacity=0.8,
                name="Barks/min",
                showlegend=False,
                text=counts,
                textposition="outside"
            ),
            row=2, col=1
        )

    # Cumulative Bark Events (row 2, col 2)
    if data.barks:
        cumulative = np.cumsum(data.barks)
        fig.add_trace(
            go.Scatter(
                x=data.timestamps,
                y=cumulative,
                mode="lines",
                name="Cumulative barks",
                line=dict(color="#e67e22", width=2),
                fill="tozeroy",
                fillcolor="rgba(230, 126, 34, 0.2)",
                showlegend=False
            ),
            row=2, col=2
        )

    # Spectrogram (row 3, col 1)
    if len(data.volumes) >= 128:
        signal = np.array(data.volumes, dtype=np.float64)
        signal -= signal.mean()
        if signal.std() > 0:
            signal /= signal.std()

        fs = max(data.sample_rate, 1)
        nperseg = min(128, len(signal) // 4)
        if nperseg >= 8:
            window = hann(nperseg)
            hop = nperseg // 2
            SFT = ShortTimeFFT(window, hop, fs, scale_to="psd")
            Z = SFT.stft(signal)
            t = SFT.t(len(signal))
            f = SFT.f

            fig.add_trace(
                go.Heatmap(
                    z=np.abs(Z),
                    x=t,
                    y=f,
                    colorscale="Magma",
                    showscale=False,
                    name="Spectrogram",
                    hovertemplate="Time: %{x:.2f}s<br>Freq: %{y:.1f}Hz<extra></extra>"
                ),
                row=3, col=1
            )

    fig.update_layout(
        height=900,
        title_text=f"Session Analysis (threshold = {threshold})",
        title_font_size=16,
        hovermode="closest",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Frequency (Hz)", row=3, col=1)

    return fig


def build_comparison_plotly(session_ids, all_data, threshold):
    """Create an interactive Plotly comparison figure."""
    n = len(all_data)
    rows = n + 2
    fig = make_subplots(
        rows=rows, cols=1,
        subplot_titles=[f"Session {sid}" for sid, _ in all_data] +
                       ["Total Bark Events per Session", "Average vs Peak Volume per Session"],
        vertical_spacing=0.06
    )

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

    for idx, (sid, data) in enumerate(all_data):
        color = colors[idx % len(colors)]
        relative_times = [(t - data.timestamps[0]).total_seconds() / 60 for t in data.timestamps]

        fig.add_trace(
            go.Scatter(
                x=relative_times,
                y=data.volumes,
                mode="lines",
                name=f"Session {sid} Volume",
                line=dict(color=color, width=1.0),
                legendgroup=f"session{sid}",
                hovertemplate="Time: %{x:.1f} min<br>Volume: %{y:.1f}<extra></extra>"
            ),
            row=idx + 1, col=1
        )

        bark_times = [
            (t - data.timestamps[0]).total_seconds() / 60
            for t, b in zip(data.timestamps, data.barks) if b
        ]
        bark_vols = [v for v, b in zip(data.volumes, data.barks) if b]
        if bark_times:
            fig.add_trace(
                go.Scatter(
                    x=bark_times,
                    y=bark_vols,
                    mode="markers",
                    name=f"Session {sid} Barks",
                    marker=dict(color=color, size=8, symbol="x"),
                    legendgroup=f"session{sid}",
                    showlegend=False,
                    hovertemplate="Time: %{x:.1f} min<br>Volume: %{y:.1f}<extra></extra>"
                ),
                row=idx + 1, col=1
            )

        duration = (data.timestamps[-1] - data.timestamps[0]).total_seconds() / 60
        avg_vol = sum(data.volumes) / len(data.volumes)
        peak_vol = max(data.volumes)
        bark_count = sum(data.barks)
        fig.update_yaxes(title_text=f"Avg {avg_vol:.0f} | Peak {peak_vol:.0f} | {bark_count} barks | {duration:.1f} min", row=idx + 1, col=1)

    # Total Barks bar chart
    labels = [f"Session {sid}" for sid, _ in all_data]
    bark_counts = [sum(data.barks) for _, data in all_data]
    fig.add_trace(
        go.Bar(
            x=labels,
            y=bark_counts,
            marker_color=colors[:n],
            opacity=0.8,
            text=bark_counts,
            textposition="outside",
            showlegend=False
        ),
        row=n + 1, col=1
    )
    fig.update_yaxes(title_text="Total Barks", row=n + 1, col=1)

    # Avg vs Peak volume
    avg_volumes = [sum(data.volumes) / len(data.volumes) for _, data in all_data]
    peak_volumes = [max(data.volumes) for _, data in all_data]
    fig.add_trace(
        go.Bar(
            x=labels,
            y=avg_volumes,
            name="Avg Volume",
            marker_color="#3498db",
            opacity=0.8,
            showlegend=True
        ),
        row=n + 2, col=1
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=peak_volumes,
            name="Peak Volume",
            marker_color="#e74c3c",
            opacity=0.8,
            showlegend=True
        ),
        row=n + 2, col=1
    )
    fig.update_yaxes(title_text="Volume (RMS)", row=n + 2, col=1)

    fig.update_layout(
        height=200 + rows * 280,
        title_text=f"Session Comparison (threshold = {threshold})",
        title_font_size=16,
        barmode="group",
        hovermode="closest",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")

    return fig


def create_plotly(log_path="bark_log.csv", threshold=1000):
    data = load_recording_data(log_path)
    if not data.timestamps:
        raise ValueError("No recording data available yet.")
    figure = build_plotly(data, threshold)
    return figure, summarize_recording(data)


def create_plotly_for_session(session_id: int, threshold=1000):
    data = load_session_data(session_id)
    if not data.timestamps:
        raise ValueError("No recording data available yet.")
    figure = build_plotly(data, threshold)
    return figure, summarize_recording(data)


def create_comparison_plotly(session_ids: list[int], threshold=1000):
    if len(session_ids) < 2:
        raise ValueError("Need at least 2 sessions to compare.")

    all_data = []
    for sid in session_ids:
        data = load_session_data(sid)
        if data.timestamps:
            all_data.append((sid, data))

    if len(all_data) < 2:
        raise ValueError("Not enough sessions with data to compare.")

    figure = build_comparison_plotly(session_ids, all_data, threshold)

    n = len(all_data)
    total_barks = sum(sum(data.barks) for _, data in all_data)
    total_duration = sum(
        (data.timestamps[-1] - data.timestamps[0]).total_seconds() / 60
        for _, data in all_data
    )
    summary = (
        f"Comparing {n} sessions | Total barks: {total_barks} | "
        f"Combined duration: {total_duration:.1f} min"
    )
    return figure, summary

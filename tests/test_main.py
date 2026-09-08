import os

import matplotlib
import numpy as np
import pytest

from analyzePlot import (
    create_comparison_plot,
    create_comparison_plotly,
    create_plot,
    create_plotly_for_session,
    load_recording_data,
    summarize_recording,
)
from database import init_db
from main import format_device_label
from record import InputDevice, compute_rms_volume

matplotlib.use("Agg")


def test_format_device_label_includes_index_and_channels():
    device = InputDevice(index=3, name="USB Mic", max_input_channels=2, default_sample_rate=44100)
    assert format_device_label(device) == "3: USB Mic (2 input ch)"


def test_load_recording_data_and_summary(tmp_path):
    log_path = tmp_path / "bark_log.csv"
    log_path.write_text(
        "timestamp,volume,is_bark\n"
        "1765101177.0,10,0\n"
        "1765101178.0,1005,1\n"
        "1765101179.0,900,0\n",
        encoding="utf-8",
    )

    data = load_recording_data(str(log_path))
    assert len(data.timestamps) == 3
    assert data.barks == [0, 1, 0]
    assert data.sample_rate > 0

    summary = summarize_recording(data)
    assert "Samples: 3" in summary
    assert "Bark events: 1" in summary


def test_create_plot_returns_matplotlib_figure(tmp_path):
    log_path = tmp_path / "bark_log.csv"
    log_path.write_text(
        "timestamp,volume,is_bark\n"
        "1765101177.0,10,0\n"
        "1765101178.0,1005,1\n",
        encoding="utf-8",
    )

    figure, summary = create_plot(str(log_path), threshold=1000)
    assert len(figure.axes) == 5
    assert "Bark events: 1" in summary


def test_create_plot_rejects_empty_data_file(tmp_path):
    log_path = tmp_path / "bark_log.csv"
    log_path.write_text("timestamp,volume,is_bark\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No recording data available yet"):
        create_plot(str(log_path), threshold=1000)


def test_compute_rms_volume_matches_int16_scale():
    samples = np.array([[0.5], [-0.5]], dtype=np.float32)
    assert compute_rms_volume(samples) == pytest.approx(16384.0, rel=1e-3)


def test_database_create_and_query(tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    assert os.path.exists(db_path)


def test_create_comparison_plot_requires_multiple_sessions():
    with pytest.raises(ValueError, match="Need at least 2 sessions"):
        create_comparison_plot([1], threshold=1000)


def test_create_comparison_plot_with_real_data(tmp_path, monkeypatch):
    db_path = str(tmp_path / "compare_test.db")
    init_db(db_path)
    import sqlite3
    conn = sqlite3.connect(db_path)
    now = 1765101177.0
    session_data = [
        (now, "Mic1", 1000.0),
        (now + 3600, "Mic2", 1000.0),
    ]
    session_ids = []
    for start_time, device_name, threshold in session_data:
        c = conn.execute(
            "INSERT INTO sessions (start_time, device_name, threshold) VALUES (?, ?, ?)",
            (start_time, device_name, threshold),
        )
        sid = c.lastrowid
        session_ids.append(sid)
        samples = []
        for i in range(5):
            t = start_time + i
            vol = 500 + i * 200
            is_bark = 1 if vol > 1000 else 0
            samples.append((sid, t, vol, is_bark))
        conn.executemany(
            "INSERT INTO samples (session_id, timestamp, volume, is_bark) VALUES (?, ?, ?, ?)",
            samples,
        )
    conn.commit()
    conn.close()

    monkeypatch.setattr("database.DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr("analyzePlot.DEFAULT_DB_PATH", db_path)

    figure, summary = create_comparison_plot(session_ids, threshold=1000)
    assert len(figure.axes) == 4  # n=2 -> 2 timeline + 1 bark bars + 1 avg/peak bars
    assert "Comparing 2 sessions" in summary


def test_create_plotly_for_session_with_data(tmp_path, monkeypatch):
    db_path = str(tmp_path / "plotly_test.db")
    init_db(db_path)
    import sqlite3
    conn = sqlite3.connect(db_path)
    now = 1765101177.0
    conn.execute(
        "INSERT INTO sessions (start_time, device_name, threshold) VALUES (?, ?, ?)",
        (now, "Mic1", 1000.0),
    )
    sid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    samples = []
    for i in range(10):
        t = now + i * 0.1
        vol = 500 + i * 100
        is_bark = 1 if vol > 1000 else 0
        samples.append((sid, t, vol, is_bark))
    conn.executemany(
        "INSERT INTO samples (session_id, timestamp, volume, is_bark) VALUES (?, ?, ?, ?)",
        samples,
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr("database.DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr("analyzePlot.DEFAULT_DB_PATH", db_path)

    figure, summary = create_plotly_for_session(sid, threshold=1000)
    assert figure is not None
    assert "Bark events:" in summary
    assert hasattr(figure, "data")


def test_create_comparison_plotly_with_real_data(tmp_path, monkeypatch):
    db_path = str(tmp_path / "plotly_compare_test.db")
    init_db(db_path)
    import sqlite3
    conn = sqlite3.connect(db_path)
    now = 1765101177.0
    session_data = [
        (now, "Mic1", 1000.0),
        (now + 3600, "Mic2", 1000.0),
    ]
    session_ids = []
    for start_time, device_name, threshold in session_data:
        c = conn.execute(
            "INSERT INTO sessions (start_time, device_name, threshold) VALUES (?, ?, ?)",
            (start_time, device_name, threshold),
        )
        sid = c.lastrowid
        session_ids.append(sid)
        samples = []
        for i in range(5):
            t = start_time + i
            vol = 500 + i * 200
            is_bark = 1 if vol > 1000 else 0
            samples.append((sid, t, vol, is_bark))
        conn.executemany(
            "INSERT INTO samples (session_id, timestamp, volume, is_bark) VALUES (?, ?, ?, ?)",
            samples,
        )
    conn.commit()
    conn.close()

    monkeypatch.setattr("database.DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr("analyzePlot.DEFAULT_DB_PATH", db_path)

    figure, summary = create_comparison_plotly(session_ids, threshold=1000)
    assert figure is not None
    assert "Comparing 2 sessions" in summary
    assert hasattr(figure, "data")


def test_detect_bark_fft_ignores_low_freq_sine_wave():
    from record import _detect_bark_fft
    # 100 Hz sine - too low for a bark, should not trigger
    t = np.linspace(0, 1, 16000, dtype=np.float32)
    samples = np.sin(2 * np.pi * 100 * t).reshape(-1, 1)
    is_bark, confidence = _detect_bark_fft(samples, 16000)
    assert is_bark == 0
    assert confidence < 0.2


def test_detect_bark_fft_triggers_on_bark_band_freq():
    from record import _detect_bark_fft
    # 800 Hz + 1200 Hz - inside bark band
    t = np.linspace(0, 1, 16000, dtype=np.float32)
    samples = (np.sin(2 * np.pi * 800 * t) + np.sin(2 * np.pi * 1200 * t) * 0.5).reshape(-1, 1)
    is_bark, confidence = _detect_bark_fft(samples, 16000)
    assert is_bark == 1
    assert confidence >= 0.35

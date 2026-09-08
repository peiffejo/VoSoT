import os
import tempfile
import tkinter as tk
import webbrowser
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ttkthemes import ThemedTk

from analyzePlot import (
    create_comparison_plotly,
    create_plotly_for_session,
)
from database import delete_session, get_sessions, init_db
from record import DEFAULT_THRESHOLD, Recorder, list_input_devices


def format_device_label(device):
    return f"{device.index}: {device.name} ({device.max_input_channels} input ch)"


class MicMeasurementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Visualasation of Sound over Time (VoSoT)")
        self.root.geometry("1800x1050+200+200")
        self.root.minsize(1200, 800)

        self.recorder = Recorder()
        self.devices = []
        self.device_lookup = {}
        self.selected_device = tk.StringVar()
        self.threshold_var = tk.StringVar(value=str(DEFAULT_THRESHOLD))
        self.status_var = tk.StringVar(value="Idle")
        self.volume_var = tk.StringVar(value="Current volume: 0.0")
        self.summary_var = tk.StringVar(value="No recording loaded yet.")
        self.figure = None
        self.canvas = None
        self.current_plotly_fig = None

        init_db()
        self._build_ui()
        self.refresh_devices()
        self.refresh_sessions()
        self._poll_status()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        controls = ttk.Frame(self.root, padding=14)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Microphone", font=("Nimbus Roman", 11, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.device_box = ttk.Combobox(controls, textvariable=self.selected_device, state="readonly", font=("Nimbus Roman", 11), width=60)
        self.device_box.grid(row=0, column=1, sticky="ew")
        ttk.Button(controls, text="Refresh devices", command=self.refresh_devices).grid(row=0, column=2, padx=(10, 0))

        ttk.Label(controls, text="Bark threshold", font=("Nimbus Roman", 11, "bold")).grid(row=1, column=0, sticky="w", pady=(12, 0), padx=(0, 10))
        ttk.Entry(controls, textvariable=self.threshold_var, font=("Nimbus Roman", 11), width=14).grid(row=1, column=1, sticky="w", pady=(12, 0))

        actions = ttk.Frame(controls)
        actions.grid(row=1, column=2, sticky="e", pady=(12, 0))
        self.start_btn = ttk.Button(actions, text="Start recording", command=self.start_recording)
        self.start_btn.grid(row=0, column=0, padx=(0, 10))
        self.stop_btn = ttk.Button(actions, text="Stop recording", command=self.stop_recording)
        self.stop_btn.grid(row=0, column=1, padx=(0, 10))
        ttk.Button(actions, text="Refresh sessions", command=self.refresh_sessions).grid(row=0, column=2)

        content = ttk.PanedWindow(self.root, orient="horizontal")
        content.grid(row=1, column=0, sticky="nsew", padx=14, pady=14)

        sidebar = ttk.Frame(content, padding=(0, 0, 10, 0))
        content.add(sidebar, weight=0)

        sessions_frame = ttk.LabelFrame(sidebar, text="Session History", padding=10)
        sessions_frame.pack(fill="both", expand=True)
        sessions_frame.columnconfigure(0, weight=1)
        sessions_frame.rowconfigure(0, weight=1)

        columns = ("id", "start", "device", "barks", "peak")
        self.session_tree = ttk.Treeview(sessions_frame, columns=columns, show="headings", height=20, selectmode="extended")
        self.session_tree.heading("id", text="ID")
        self.session_tree.heading("start", text="Start")
        self.session_tree.heading("device", text="Device")
        self.session_tree.heading("barks", text="Barks")
        self.session_tree.heading("peak", text="Peak vol")
        self.session_tree.column("id", width=40, anchor="center")
        self.session_tree.column("start", width=160, anchor="center")
        self.session_tree.column("device", width=180)
        self.session_tree.column("barks", width=60, anchor="center")
        self.session_tree.column("peak", width=80, anchor="center")
        self.session_tree.grid(row=0, column=0, sticky="nsew")
        self.session_tree.bind("<<TreeviewSelect>>", self._on_session_select)

        sb = ttk.Scrollbar(sessions_frame, orient="vertical", command=self.session_tree.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.session_tree.configure(yscrollcommand=sb.set)

        session_actions = ttk.Frame(sidebar, padding=(0, 10, 0, 0))
        session_actions.pack(fill="x")
        ttk.Button(session_actions, text="Load selected session", command=self._load_selected_session).pack(side="left", padx=(0, 10))
        ttk.Button(session_actions, text="Delete selected session", command=self._delete_selected_session).pack(side="left", padx=(0, 10))
        ttk.Button(session_actions, text="Compare sessions", command=self._compare_selected_sessions).pack(side="left")

        main_area = ttk.Frame(content)
        content.add(main_area, weight=1)
        main_area.columnconfigure(0, weight=1)
        main_area.rowconfigure(1, weight=1)

        status_frame = ttk.LabelFrame(main_area, text="Status", padding=12)
        status_frame.grid(row=0, column=0, sticky="ew")
        status_frame.columnconfigure(0, weight=1)
        ttk.Label(status_frame, textvariable=self.status_var, font=("Nimbus Roman", 10)).grid(row=0, column=0, sticky="w")
        ttk.Label(status_frame, textvariable=self.volume_var, font=("Nimbus Roman", 10)).grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.volume_meter = ttk.Progressbar(status_frame, mode="determinate", maximum=100, length=400)
        self.volume_meter.grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Label(status_frame, textvariable=self.summary_var, wraplength=1000, font=("Noto Mono", 10)).grid(row=3, column=0, sticky="w", pady=(10, 0))

        plot_frame = ttk.LabelFrame(main_area, text="Visualization", padding=12)
        plot_frame.grid(row=1, column=0, sticky="nsew", pady=(14, 0))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        self.plot_frame = plot_frame
        self.empty_plot_label = ttk.Label(plot_frame, text="Start a recording or select a session from the sidebar to see the charts.", font=("Noto Mono", 9))
        self.empty_plot_label.grid(row=0, column=0)

    def refresh_devices(self):
        self.devices = list_input_devices()
        labels = [format_device_label(device) for device in self.devices]
        self.device_lookup = {label: device for label, device in zip(labels, self.devices)}
        self.device_box["values"] = labels

        if labels:
            current = self.selected_device.get()
            if current in self.device_lookup:
                self.selected_device.set(current)
            else:
                # Prefer the system default device, then pipewire/pulse, then first device
                preferred = None
                for device in self.devices:
                    if device.name.strip().lower() == "default":
                        preferred = format_device_label(device)
                        break
                if preferred is None:
                    for device in self.devices:
                        if device.name.strip().lower() in ("pipewire", "pulse"):
                            preferred = format_device_label(device)
                            break
                self.selected_device.set(preferred if preferred else labels[0])
            self.status_var.set("Ready!")
        else:
            self.selected_device.set("")
            self.status_var.set("No input devices found!")

    def refresh_sessions(self):
        for item in self.session_tree.get_children():
            self.session_tree.delete(item)
        sessions = get_sessions()
        for session in sessions:
            start_str = session.start_time.strftime("%Y-%m-%d %H:%M")
            self.session_tree.insert(
                "",
                "end",
                iid=str(session.id),
                values=(
                    session.id,
                    start_str,
                    session.device_name,
                    session.bark_count,
                    f"{session.peak_volume:.1f}",
                ),
            )

    def _on_session_select(self, event):
        del event

    def _get_selected_session_id(self):
        selection = self.session_tree.selection()
        if not selection:
            return None
        return int(selection[0])

    def _load_selected_session(self):
        session_id = self._get_selected_session_id()
        if session_id is None:
            messagebox.showwarning("No session selected", "Select a session from the sidebar first.")
            return
        try:
            threshold = self._parse_threshold()
        except FileNotFoundError as exc:
            messagebox.showerror("File not found", str(exc))
            return
        except BaseException:
            raise
        try:
            figure, summary = create_plotly_for_session(session_id, threshold=int(threshold))
        except ValueError as exc:
            self.summary_var.set(str(exc))
            self._clear_plot()
            return

        self.summary_var.set(summary)
        self._show_plotly(figure)

    def _delete_selected_session(self):
        session_id = self._get_selected_session_id()
        if session_id is None:
            messagebox.showwarning("No session selected", "Select a session from the sidebar first.")
            return
        if messagebox.askyesno("Delete session", f"Delete session {session_id}? This cannot be undone."):
            delete_session(session_id)
            self.refresh_sessions()

    def _compare_selected_sessions(self):
        selection = self.session_tree.selection()
        if len(selection) < 2:
            messagebox.showwarning("Not enough sessions", "Select at least 2 sessions in the sidebar to compare.")
            return
        if len(selection) > 5:
            messagebox.showwarning("Too many sessions", "Please select max 5 sessions for comparison.")
            return

        session_ids = [int(sid) for sid in selection]
        try:
            threshold = self._parse_threshold()
        except ValueError as exc:
            messagebox.showerror("Invalid threshold", str(exc))
            return

        try:
            figure, summary = create_comparison_plotly(session_ids, threshold=int(threshold))
        except ValueError as exc:
            self.summary_var.set(str(exc))
            self._clear_plot()
            return

        self.summary_var.set(summary)
        self._show_plotly(figure)

    def _parse_threshold(self):
        try:
            threshold = float(self.threshold_var.get())
        except ValueError as exc:
            raise ValueError("Threshold must be a number.") from exc

        if threshold <= 0:
            raise ValueError("Threshold must be greater than 0.")
        return threshold

    def start_recording(self):
        label = self.selected_device.get()
        if label not in self.device_lookup:
            messagebox.showerror("No microphone selected", "Pick an input device before starting a recording.")
            return

        try:
            threshold = int(self._parse_threshold())
        except ValueError as exc:
            messagebox.showerror("Invalid threshold", str(exc))
            return

        self.recorder.threshold = threshold
        device = self.device_lookup[label]
        started = self.recorder.start(
            device.index,
            device_name=device.name,
            channels=1,
            rate=device.default_sample_rate,
        )
        if not started:
            messagebox.showinfo("Recording already running", "Stop the current recording before starting a new one.")
            return

        self.status_var.set(f"Recording from {device.name}")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

    def stop_recording(self):
        stopped = self.recorder.stop()
        if stopped:
            self.status_var.set("Recording stopped")
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self.refresh_sessions()
            self._load_latest_session()

    def _load_latest_session(self):
        sessions = get_sessions()
        if not sessions:
            return
        latest = sessions[0]
        try:
            threshold = self._parse_threshold()
        except ValueError:
            threshold = DEFAULT_THRESHOLD
        try:
            figure, summary = create_plotly_for_session(latest.id, threshold=int(threshold))
            self.summary_var.set(summary)
            self._show_plotly(figure)
        except ValueError:
            pass

    def _show_figure(self, figure):
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        if self.figure is not None:
            self.figure.clear()

        self.figure = figure
        self.empty_plot_label.grid_remove()
        self.canvas = FigureCanvasTkAgg(figure, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _show_plotly(self, figure):
        """Display a Plotly figure by opening it in the default web browser."""
        self.current_plotly_fig = figure
        html_path = os.path.join(tempfile.gettempdir(), "mic_measurement_plot.html")
        figure.write_html(html_path, include_plotlyjs="cdn")
        webbrowser.open(f"file:///{html_path.replace(os.sep, '/')}")

    def _clear_plot(self):
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        self.empty_plot_label.grid()

    def _poll_status(self):
        status = self.recorder.get_status()
        conf = status.get("bark_confidence", 0.0)
        self.volume_var.set(
            f"Current volume: {status['latest_volume']:.1f} "
            f"Confidence: {conf:.1%} | "
        )
        try:
            threshold = max(float(self.threshold_var.get() or DEFAULT_THRESHOLD), 1.0)
        except ValueError:
            threshold = float(DEFAULT_THRESHOLD)
        self.volume_meter["value"] = min((status["latest_volume"] / threshold) * 100, 100)

        if status["last_error"]:
            self.status_var.set(f"Recorder error: {status['last_error']}")

        self.root.after(300, self._poll_status)

    def on_close(self):
        self.recorder.stop()
        self.root.destroy()


def main():
    root = ThemedTk(theme="arc")
    MicMeasurementApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

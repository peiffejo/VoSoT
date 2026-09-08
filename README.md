# MicMeasurement

A small desktop application to visualize microphone volume over time and detect bark-like events.

## Features

- Select an input device from a dropdown
- Record microphone input with a configurable bark threshold
- Visualize recorded sessions with Plotly charts
- Compare multiple sessions side-by-side
- SQLite-backed session history

## Requirements

- Python 3.12
- Linux (the bundled executable below is built for 64-bit Linux)
- A working microphone / audio input device

## Run from source

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
python main.py
```

## Download

### Latest release

The easiest way to get the app is from the [Releases](https://github.com/peiffejo/VoSoT/releases/latest) page.

| Platform | Asset | Notes |
|----------|-------|-------|
| Linux 64-bit | `MicMeasurement` | Single executable, built with PyInstaller |

After downloading, make it executable and run it from a terminal:

```bash
chmod +x MicMeasurement
./MicMeasurement
```

### Run from source

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
python main.py
```

## Project structure

```
MicMeasurement/
├── main.py              # Tkinter application entry point
├── record.py            # Audio recording and bark detection
├── analyzePlot.py       # Plotly / Matplotlib visualization helpers
├── database.py          # SQLite session storage
├── requirements.txt     # Python dependencies
├── main.spec            # PyInstaller spec for Linux builds
└── tests/               # Unit tests
```

## License

MIT

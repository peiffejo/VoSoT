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

### Latest build

| Platform | File | Size |
|----------|------|------|
| Linux 64-bit | [`dist/MicMeasurement`](./dist/MicMeasurement) | ~90 MB |

> The Linux build was generated with PyInstaller. It is a single executable file.
> To run it from the terminal:
>
> ```bash
> ./dist/MicMeasurement
> ```

### GitHub Releases (recommended)

For pre-built binaries for Windows, macOS and Linux, see the [Releases](https://github.com/DEIN_USERNAME/MicMeasurement/releases) page.

Replace `DEIN_USERNAME` with your GitHub username once the repository is published.

## Build from source with PyInstaller

```bash
python -m PyInstaller main.spec
```

The output is written to `dist/MicMeasurement`.

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

# 🎸 Chord Detector

Real-time audio chord detection using your microphone. Available as a command-line tool and a web application with live visualizations.

## Features

- **🎵 Multi-pitch Detection** — Detects multiple notes simultaneously for accurate chord recognition
- **🎸 10 Instrument Presets** — Optimized frequency ranges for guitar, piano, bass, violin, cello, flute, clarinet, saxophone, trumpet, and voice
- **📊 Chroma Vector Analysis** — Uses 12-bin chroma representation for robust chord matching
- **🌐 Web Interface** — Stream audio from your browser with real-time visualizations
- **⚙️ Configurable** — Sensitivity, thresholds, overlap ratio, and confidence levels
- **📈 Chord Progression** — Tracks chord stability and progression over time

## Installation

### Prerequisites

- Python 3.8+
- Working microphone

### Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Core dependencies
pip install numpy scipy sounddevice

# For web interface
pip install -r requirements/requirements_web.txt

# For running tests
pip install pytest pytest-asyncio
```

## Quick Start

### Command-Line Tool

```bash
# Basic usage (guitar, default settings)
python chord_detector.py

# With logging
python chord_detector.py --log

# Different instrument
python chord_detector.py --instrument piano --log
```

### Web Interface

```bash
# Start the server
python web_server.py

# Open in browser
# http://localhost:9103/
```

---

## Command-Line Interface

### Basic Usage

```bash
python chord_detector.py [OPTIONS]
```

### Common Options

| Option | Default | Description |
|--------|---------|-------------|
| `--instrument` | `guitar` | Instrument preset (guitar, piano, bass, etc.) |
| `--log` | off | Enable timestamped output |
| `--debug` | off | Show detailed debug information |
| `--confidence-threshold` | `0.6` | Minimum confidence to display chords (0.0-1.0) |
| `--sensitivity` | `1.0` | Detection sensitivity (0.1-2.0) |
| `--silence-threshold` | `0.005` | Audio level threshold |
| `--overlap` | `0.75` | Window overlap ratio (0.0-0.9) |

### Examples

```bash
# Guitar with chord logging
python chord_detector.py --log --show-chroma

# Piano with lower confidence threshold
python chord_detector.py --instrument piano --confidence-threshold 0.4 --log

# Bass with higher sensitivity
python chord_detector.py --instrument bass --sensitivity 1.5 --log

# Debug mode to tune thresholds
python chord_detector.py --debug

# Custom frequency range
python chord_detector.py --low-freq 100 --high-freq 3000 --log
```

### Output Modes

```bash
# Show detected frequencies
python chord_detector.py --show-frequencies

# Show only notes (no chord analysis)
python chord_detector.py --notes-only

# Show only frequencies (skip chord processing)
python chord_detector.py --frequencies-only

# Show chroma vector with chords
python chord_detector.py --show-chroma --log
```

### All Options

```bash
python chord_detector.py --help
```

---

## Web Interface

### Starting the Server

```bash
# Default (port 9103)
python web_server.py

# Custom port
python web_server.py --port 8080

# With debug logging
python web_server.py --log-level DEBUG

# Multiple workers
python web_server.py --workers 4
```

### Accessing the Interface

Open your browser to:

```
http://localhost:9103/
```

### URL Parameters

Configure the detector via URL parameters:

```
http://localhost:9103/?instrument=piano&confidence_threshold=0.5&debug=true
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `instrument` | `guitar` | Instrument preset |
| `sensitivity` | `1` | Detection sensitivity |
| `confidence_threshold` | `0.6` | Minimum chord confidence |
| `silence_threshold` | `0.005` | Audio silence threshold |
| `overlap` | `0.75` | Window overlap ratio |
| `show_chroma` | `false` | Show chroma vector visualization |
| `debug` | `false` | Enable debug logging |
| `log` | `false` | Enable timestamped logging |

### Web Interface Features

1. **Start Detection** — Begin streaming audio from your microphone
2. **Visualizations** — Real-time frequency spectrum, chroma vector, and waveform
3. **Settings Panel** — Adjust all parameters without reloading
4. **Status Indicator** — Shows connection and recording status

---

## Supported Instruments

| Instrument | Frequency Range | Best For |
|------------|-----------------|----------|
| **Guitar** | 80–2000 Hz | Acoustic/Electric guitar |
| **Piano** | 100–4000 Hz | Piano, keyboard |
| **Bass** | 40–800 Hz | Bass guitar, double bass |
| **Violin** | 200–3000 Hz | Violin, fiddle |
| **Cello** | 65–1000 Hz | Cello |
| **Flute** | 250–2500 Hz | Flute, piccolo |
| **Clarinet** | 150–1500 Hz | Clarinet |
| **Saxophone** | 100–800 Hz | Saxophone |
| **Trumpet** | 150–1000 Hz | Trumpet, brass |
| **Voice** | 80–1000 Hz | Vocal detection |

---

## Recognized Chords

The detector recognizes these chord types:

- **Triads**: Major, Minor, Diminished, Augmented
- **7th Chords**: Major 7th, Minor 7th, Dominant 7th
- **6th Chords**: Major 6th, Minor 6th
- **Suspended**: Sus2, Sus4
- **Power Chords**: Root + Fifth

Output format: `C`, `Am`, `G7`, `Dm7`, `Fsus4`, `E5`, etc.

---

## Project Structure

```
music_test/
├── chord_detector.py      # CLI application
├── web_server.py          # FastAPI web server
├── pytest.ini             # Test configuration
├── lib/
│   ├── common.py          # Shared constants and utilities
│   ├── config.py          # Unified configuration interface
│   ├── state.py           # Audio processing state management
│   ├── output.py          # Output handlers (console/dict)
│   ├── audio_processing.py    # Core audio processing (shared)
│   ├── music_understanding.py # Chord detection algorithms
│   ├── sound_capture.py   # CLI audio capture wrapper
│   └── web_audio_processing.py # Web audio processing wrapper
├── tests/
│   ├── conftest.py        # Pytest fixtures
│   ├── test_common.py     # Tests for common.py
│   ├── test_music_understanding.py  # Chord detection tests
│   ├── test_audio_processing.py     # Audio processing tests
│   ├── test_integration.py          # End-to-end tests
│   └── test_unified_classes.py      # Tests for unified classes
├── static/
│   ├── css/style.css      # Web interface styles
│   └── js/
│       ├── main.js        # Main application logic
│       ├── visualizations.js  # Canvas visualizations
│       └── audio_handler.js   # Audio capture utilities
├── templates/
│   └── index.html         # Web interface template
└── requirements/
    └── requirements_web.txt   # Web server dependencies
```

## Architecture

The CLI and web interfaces share a unified core processing function:

```
                    ┌─────────────────────────────┐
                    │  lib/audio_processing.py    │
                    │  process_audio_chunk()      │
                    │  (Core Logic)               │
                    └─────────────┬───────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
    ┌───────▼───────┐     ┌───────▼───────┐     ┌───────▼───────┐
    │ lib/config.py │     │ lib/state.py  │     │ lib/output.py │
    │ AudioConfig   │     │ AudioProcess- │     │ Console/Dict  │
    │               │     │ ingState      │     │ OutputHandler │
    └───────────────┘     └───────────────┘     └───────────────┘
            │                     │                     │
    ┌───────┴─────────────────────┴─────────────────────┴───────┐
    │                         Consumers                          │
    ├────────────────────────┬──────────────────────────────────┤
    │ lib/sound_capture.py   │  lib/web_audio_processing.py     │
    │ recognize_audio()      │  recognize_audio_web()           │
    │ (CLI)                  │  (Web)                           │
    └────────────────────────┴──────────────────────────────────┘
```

## Running Tests

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_music_understanding.py -v

# Run with coverage
python -m pytest tests/ --cov=lib --cov-report=html
```

---

## Troubleshooting

### No Audio Detected

1. Check microphone permissions in your OS
2. Ensure no other application is using the microphone
3. Try increasing `--sensitivity` (e.g., `--sensitivity 1.5`)
4. Use `--debug` to see audio levels

### Wrong Chords Detected

1. Lower `--confidence-threshold` (e.g., `0.4`)
2. Select the correct `--instrument` preset
3. Reduce background noise
4. Adjust `--silence-threshold` if too sensitive

### Web Interface Issues

1. **"Microphone Error"** — Grant microphone permission in browser
2. **"Connection Timeout"** — Ensure server is running
3. **HTTPS Required** — Some browsers require HTTPS for microphone access on non-localhost

### Debug Mode

```bash
# CLI debug
python chord_detector.py --debug

# Web server debug
python web_server.py --log-level DEBUG
```

---

## License

MIT License


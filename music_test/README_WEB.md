# Chord Detector Web Interface

A real-time chord detection web application that streams audio from your browser and displays chord detection results with visualizations.

## Features

- 🎸 Real-time chord detection from browser microphone
- 📊 Visual frequency spectrum display
- 🌈 Chroma vector visualization
- 🎵 Detected notes display
- ⚙️ Configurable settings (instrument, sensitivity, thresholds)
- 📝 Server-side logging mode
- 🔍 Debug mode for troubleshooting

## Installation

### 1. Install Python Dependencies

```bash
# Install base dependencies (if not already installed)
pip install numpy sounddevice scipy

# Install web server dependencies
pip install -r requirements/requirements_web.txt
```

### 2. Directory Structure

Ensure you have the following directory structure:
```
music_test/
├── web_server.py
├── lib/
│   ├── common.py              # Shared constants
│   ├── config.py              # Unified configuration interface
│   ├── state.py               # Audio processing state
│   ├── output.py              # Output handlers
│   ├── audio_processing.py    # Core processing (shared with CLI)
│   ├── music_understanding.py # Chord detection algorithms
│   ├── sound_capture.py       # CLI audio capture
│   └── web_audio_processing.py # Web audio processing wrapper
├── templates/
│   └── index.html
├── static/
│   ├── js/
│   │   ├── main.js
│   │   ├── visualizations.js
│   │   └── audio_handler.js
│   └── css/
│       └── style.css
└── requirements/
    └── requirements_web.txt
```

## Running the Server

### Basic Usage

```bash
python web_server.py
```

This starts the server on `http://0.0.0.0:9103` by default.

### Server Options

```bash
python web_server.py --help
```

Available options:
- `--host HOST`: Host to bind to (default: `0.0.0.0`)
- `--port PORT`: Port to bind to (default: `9103`)
- `--reload`: Enable auto-reload for development
- `--log-level LEVEL`: Logging level - DEBUG, INFO, WARNING, ERROR, CRITICAL (default: `INFO`)
- `--workers N`: Number of worker processes (default: `1`)

### Examples

```bash
# Run on custom port
python web_server.py --port 8080

# Run on specific host
python web_server.py --host 127.0.0.1 --port 9103

# Development mode with auto-reload
python web_server.py --reload

# Debug logging
python web_server.py --log-level DEBUG

# Multiple workers for production
python web_server.py --workers 4
```

> **Note**: `--reload` cannot be used with `--workers > 1`.

## Accessing the Web Interface

### Direct Access

Open your browser and navigate to:
```
http://localhost:9103/
```

### Through Envoy Proxy

If using Envoy proxy (as configured in `http-proxy/config/envoy.yaml`):
```
http://your-envoy-host:8888/chord_detector/
```

## URL Parameters

You can configure the detector via URL query parameters:

### Basic Parameters

- `instrument`: Instrument preset (default: `guitar`)
  - Options: `guitar`, `piano`, `bass`, `violin`, `cello`, `flute`, `clarinet`, `saxophone`, `trumpet`, `voice`
  - Example: `?instrument=piano`

- `sensitivity`: Detection sensitivity (default: `1.0`, range: `0.1-2.0`)
  - Example: `?sensitivity=1.5`

- `confidence_threshold`: Minimum confidence to show chords (default: `0.6`, range: `0.1-1.0`)
  - Example: `?confidence_threshold=0.7`

- `silence_threshold`: Silence detection threshold (default: `0.005`, range: `0.001-0.1`)
  - Example: `?silence_threshold=0.01`

- `overlap`: Overlap ratio for audio processing (default: `0.75`, range: `0.0-0.9`)
  - Example: `?overlap=0.8`

### Advanced Parameters

- `progression`: Enable chord progression analysis (default: `true`)
  - Example: `?progression=true`

- `multi_pitch`: Enable multi-pitch detection (default: `true`)
  - Example: `?multi_pitch=false` (uses single-pitch autocorrelation)

- `show_frequencies`: Show detected frequencies (default: `false`)
  - Example: `?show_frequencies=true`

- `show_chroma`: Show chroma vector (default: `false`)
  - Example: `?show_chroma=true`

- `debug`: Enable debug mode with server logging (default: `false`)
  - Example: `?debug=true`

- `log`: Enable logging mode with timestamps (default: `false`)
  - Example: `?log=true`

### Custom Frequency Range

- `low_freq`: Custom low frequency cutoff in Hz
- `high_freq`: Custom high frequency cutoff in Hz
  - Example: `?low_freq=100&high_freq=3000`

### Complete URL Example

```
http://localhost:9103/?instrument=guitar&sensitivity=1.2&confidence_threshold=0.7&show_frequencies=true&show_chroma=true&debug=true
```

## Using the Web Interface

1. **Start the Server**: Run `python web_server.py`

2. **Open Browser**: Navigate to `http://localhost:9103/`

3. **Grant Microphone Permission**: When prompted, allow microphone access

4. **Click "Start Detection"**: Begin streaming audio to the server

5. **View Results**: 
   - Chord name appears in large display
   - Detected notes shown below
   - Visualizations update in real-time

6. **Adjust Settings**: Click "Settings" button to modify detection parameters

7. **Stop Detection**: Click "Stop Detection" when done

## Server Logging

### Debug Mode

When `debug=true` is set (via URL parameter or settings), the server logs:
- Audio level readings
- Detected notes
- Chord analysis results
- Confidence scores
- Error messages

Example output:
```
[2024-01-15 10:30:45] 🎸 New connection: 127.0.0.1:12345 (Guitar)
[2024-01-15 10:30:46] Audio Level: 0.0234 (threshold: 0.005)
[2024-01-15 10:30:46] detected_notes: ['C', 'E', 'G']
[2024-01-15 10:30:46] chord analysis: C Major, confidence: 0.850
[2024-01-15 10:30:46] [127.0.0.1:12345] C Major (0.85) [stable]
```

### Log Mode

When `log=true` is set, the server logs timestamped detections:
```
[2024-01-15 10:30:45] [127.0.0.1:12345] Notes: C(262Hz), E(330Hz), G(392Hz) -> C Major
[2024-01-15 10:30:46] [127.0.0.1:12345] Notes: C(262Hz), E(330Hz), G(392Hz) -> C Major
```

### Combined Debug + Log Mode

Both modes can be enabled simultaneously for comprehensive logging:
```
?debug=true&log=true
```

## Envoy Proxy Configuration

The Envoy configuration in `http-proxy/config/envoy.yaml` includes:

- **WebSocket endpoint**: `/chord_detector/ws` → routes to port 9103
- **Web interface**: `/chord_detector/` → routes to port 9103
- **Static files**: `/chord_detector/static/` → routes to port 9103
- **Health check**: `/chord_detector/health` → routes to port 9103

### Short Alias

A short alias `/c_d/` is available for convenience:

- `https://example.com/c_d/` → same as `/chord_detector/`
- `https://example.com/c_d/ws` → same as `/chord_detector/ws`

The service is configured to run on `host.docker.internal:9103` (adjust if needed).

## Troubleshooting

### Microphone Not Working

1. **Check Browser Permissions:**
   - Click the lock/info icon in your browser's address bar
   - Ensure microphone access is set to "Allow"
   - If blocked, change to "Allow" and refresh the page

2. **HTTPS Requirement:**
   - If accessing via IP address (e.g., `192.168.x.x:9103`), some browsers require HTTPS
   - Try accessing via `localhost:9103` instead
   - Or set up HTTPS/SSL for your server

3. **Browser Compatibility:**
   - Use a modern browser (Chrome, Firefox, Edge, Safari)
   - Ensure browser is up to date

4. **Check Microphone:**
   - Ensure microphone is not being used by another application
   - Test microphone in another application first
   - Check system microphone settings

5. **Common Error Messages:**
   - **"Permission denied"**: Grant microphone permission in browser settings
   - **"No microphone found"**: Connect a microphone device
   - **"Already in use"**: Close other applications using the microphone
   - **"Not supported"**: Use a modern browser or enable HTTPS

### No Chords Detected

1. Increase `sensitivity` parameter (try `1.5` or `2.0`)
2. Lower `confidence_threshold` (try `0.4` or `0.5`)
3. Check `silence_threshold` - may be too high
4. Enable `debug=true` to see what's being detected
5. Try different instrument preset

### WebSocket Connection Issues

1. Check server is running: `curl http://localhost:9103/health`
2. Check firewall settings
3. If using Envoy, verify proxy configuration
4. Check browser console for errors

### Server Errors

1. Enable `debug=true` for detailed error messages
2. Check Python dependencies are installed
3. Verify audio processing libraries (numpy, scipy) are working
4. Check server logs for exceptions

## Development

### Running in Development Mode

```bash
python web_server.py --reload
```

This enables auto-reload when code changes.

### Running Tests

The project includes a comprehensive test suite:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests
python -m pytest tests/ -v

# Run only web-related tests
python -m pytest tests/test_integration.py -v

# Run unified classes tests
python -m pytest tests/test_unified_classes.py -v
```

### Testing WebSocket Connection

You can test the WebSocket endpoint directly:

```python
import asyncio
import websockets
import json

async def test():
    uri = "ws://localhost:9103/ws"
    async with websockets.connect(uri) as websocket:
        config = {
            "instrument": "guitar",
            "sensitivity": 1.0,
            "confidence_threshold": 0.6
        }
        await websocket.send(json.dumps(config))
        response = await websocket.recv()
        print(f"Response: {response}")

asyncio.run(test())
```

### Testing Core Processing

You can test the core audio processing function directly:

```python
import numpy as np
from lib.audio_processing import process_audio_chunk
from lib.state import AudioProcessingState

# Create test audio (440Hz sine wave)
sample_rate = 44100
duration = 0.1
t = np.linspace(0, duration, int(sample_rate * duration))
audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

# Create state and config
config = {'instrument': 'guitar', 'sensitivity': 1.0}
state = AudioProcessingState(config)

# Process audio
result = process_audio_chunk(audio, state, config)
print(result)
```

## Architecture

### Overview

- **Frontend**: HTML/JavaScript with Web Audio API for microphone capture
- **Backend**: FastAPI with WebSocket support for real-time audio streaming
- **Processing**: Shared core processing function used by both CLI and web
- **State Management**: Per-connection state (buffers, history, stability tracking)

### Unified Core Architecture

The web interface shares the same core audio processing logic as the CLI:

```
┌─────────────────────────────────────────────────────────────┐
│                    Core Processing Layer                     │
├─────────────────────────────────────────────────────────────┤
│  lib/audio_processing.py::process_audio_chunk()             │
│  - Buffer management                                         │
│  - Note detection via FFT/autocorrelation                   │
│  - Chord analysis (progression, enhanced matching)          │
│  - Confidence scoring                                        │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
   ┌──────────▼──────────┐         ┌──────────▼──────────┐
   │  CLI Interface      │         │  Web Interface      │
   │  sound_capture.py   │         │  web_audio_         │
   │  - sounddevice      │         │  processing.py      │
   │  - Console output   │         │  - WebSocket input  │
   │                     │         │  - JSON output      │
   └─────────────────────┘         └─────────────────────┘
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `lib/config.py` | Unified configuration wrapper (handles both argparse and dict) |
| `lib/state.py` | Audio processing state (buffers, history, stability) |
| `lib/output.py` | Output handlers (ConsoleOutputHandler, DictOutputHandler) |
| `lib/audio_processing.py` | Core `process_audio_chunk()` function |
| `lib/web_audio_processing.py` | Web-specific wrapper with logging |

### Benefits

- **Consistent behavior**: CLI and web produce identical detection results
- **Single source of truth**: Bug fixes and improvements apply to both interfaces
- **Testable**: Core function can be unit tested independently
- **Configurable output**: Same processing, different output formats

## API Endpoints

- `GET /`: Main web interface
- `GET /health`: Health check endpoint
- `WebSocket /ws`: Audio streaming and chord detection

## License

Same as the main chord detector project.


"""
FastAPI web server for chord detection with WebSocket audio streaming.
"""
import asyncio
import json
import os
import time
import numpy as np
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional
import argparse
import sys
from lib.common import get_hop_size, get_overlap_ratio, set_overlap_ratio, get_buffer_size, get_chunk, get_rate, get_channels
from lib.music_understanding import INSTRUMENT_PRESETS
from lib.web_audio_processing import recognize_audio_web

app = FastAPI(title="Chord Detector Web Service")

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    pass  # Static directory might not exist yet

# Per-connection state
class ConnectionState:
    def __init__(self, config):
        self.audio_buffer = np.zeros(get_buffer_size(), dtype=np.float32)
        self.buffer_index = 0
        self.notes_history = deque(maxlen=5)
        self.frequencies_history = deque(maxlen=5)
        self.chroma_history = deque(maxlen=5)
        self.last_chord = None
        self.chord_stability = 0
        self.last_log_time = time.time()
        self.config = config
        self.low_freq = config.get('low_freq', 80)
        self.high_freq = config.get('high_freq', 2000)
        self.instrument_name = config.get('instrument_name', 'Guitar')

# Store active connections
active_connections: dict[str, ConnectionState] = {}

def parse_config_from_query(query_params: dict) -> dict:
    """Parse configuration from query parameters - match CLI defaults exactly."""
    config = {
        'sensitivity': float(query_params.get('sensitivity', 1.0)),
        'silence_threshold': float(query_params.get('silence_threshold', 0.005)),  # Match CLI default
        'low_freq': None,
        'high_freq': None,
        'instrument_name': None,
        'instrument': query_params.get('instrument', 'guitar'),
        'overlap': float(query_params.get('overlap', 0.75)),  # Match CLI default
        'show_frequencies': query_params.get('show_frequencies', 'false').lower() == 'true',
        'show_fft': query_params.get('show_fft', 'false').lower() == 'true',
        'raw_frequencies': query_params.get('raw_frequencies', 'false').lower() == 'true',
        'frequencies_only': query_params.get('frequencies_only', 'false').lower() == 'true',
        'notes_only': query_params.get('notes_only', 'false').lower() == 'true',
        'show_chroma': query_params.get('show_chroma', 'false').lower() == 'true',
        'single_pitch': query_params.get('single_pitch', 'false').lower() == 'true',
        'multi_pitch': query_params.get('multi_pitch', 'true').lower() == 'true',  # Match CLI default
        'confidence_threshold': float(query_params.get('confidence_threshold', 0.6)),  # Match CLI default
        'progression': query_params.get('progression', 'true').lower() == 'true',  # Match CLI default
        'debug': query_params.get('debug', 'false').lower() == 'true',
        'log': query_params.get('log', 'false').lower() == 'true',
        'log_interval': float(query_params.get('log_interval', 0.5)),  # Match CLI default
        'amplitude_threshold': float(query_params.get('amplitude_threshold', 0.005)),  # Match CLI default (though not used in processing)
    }
    
    # Handle single-pitch override
    if config['single_pitch']:
        config['multi_pitch'] = False
    
    # Get instrument settings
    if query_params.get('low_freq') and query_params.get('high_freq'):
        config['low_freq'] = int(query_params.get('low_freq'))
        config['high_freq'] = int(query_params.get('high_freq'))
        config['instrument_name'] = f"Custom ({config['low_freq']}-{config['high_freq']} Hz)"
    else:
        preset = INSTRUMENT_PRESETS.get(config['instrument'], INSTRUMENT_PRESETS['guitar'])
        config['low_freq'] = preset['low_freq']
        config['high_freq'] = preset['high_freq']
        config['instrument_name'] = preset['name']
    
    # Update overlap settings
    set_overlap_ratio(config['overlap'])
    
    return config

@app.get("/")
async def get_index():
    """Serve the main HTML page."""
    try:
        return FileResponse("templates/index.html")
    except Exception:
        # Fallback HTML if file doesn't exist
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head><title>Chord Detector</title></head>
        <body>
            <h1>Chord Detector Web Interface</h1>
            <p>Please ensure templates/index.html exists</p>
        </body>
        </html>
        """)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "connections": len(active_connections)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for audio streaming and chord detection."""
    await websocket.accept()
    connection_id = f"{websocket.client.host}:{id(websocket)}"
    
    try:
        # Receive initial configuration
        config_msg = await websocket.receive_text()
        config = json.loads(config_msg)
        
        # Set overlap ratio BEFORE creating state or printing (matches CLI behavior)
        set_overlap_ratio(config.get('overlap', 0.75))
        
        # Create connection state
        state = ConnectionState(config)
        active_connections[connection_id] = state
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🎸 New connection: {connection_id} ({state.instrument_name})")
        
        # Only print detailed configuration in debug mode (client debug OR server log level DEBUG)
        if config.get('debug') or SERVER_LOG_LEVEL == 'DEBUG':
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 📊 Configuration Parameters:")
            print(f"  Instrument: {config.get('instrument', 'guitar')} ({state.instrument_name})")
            print(f"  Frequency Range: {state.low_freq}-{state.high_freq} Hz")
            print(f"  Sensitivity: {config.get('sensitivity', 1.0)}")
            print(f"  Confidence Threshold: {config.get('confidence_threshold', 0.6)}")
            print(f"  Silence Threshold: {config.get('silence_threshold', 0.005)}")
            print(f"  Amplitude Threshold: {config.get('amplitude_threshold', 0.005)}")
            print(f"  Overlap: {config.get('overlap', 0.75)} ({get_overlap_ratio()*100:.0f}%, hop size: {get_hop_size()} samples)")
            print(f"  Progression: {config.get('progression', True)}")
            print(f"  Multi-pitch: {config.get('multi_pitch', True)}")
            print(f"  Single-pitch: {config.get('single_pitch', False)}")
            print(f"  Show Frequencies: {config.get('show_frequencies', False)}")
            print(f"  Show Chroma: {config.get('show_chroma', False)}")
            print(f"  Show FFT: {config.get('show_fft', False)}")
            print(f"  Raw Frequencies: {config.get('raw_frequencies', False)}")
            print(f"  Frequencies Only: {config.get('frequencies_only', False)}")
            print(f"  Notes Only: {config.get('notes_only', False)}")
            print(f"  Debug: {config.get('debug', False)}")
            print(f"  Log: {config.get('log', False)}")
            print(f"  Log Interval: {config.get('log_interval', 0.5)}s")
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 📊 Full Config JSON: {json.dumps(config, indent=2)}")
        
        # Send initial acknowledgment
        await websocket.send_json({
            "type": "connected",
            "message": f"Connected for {state.instrument_name}",
            "config": config
        })
        
        # Main processing loop
        audio_chunk_count = 0
        while True:
            try:
                # Receive audio data (binary)
                message = await websocket.receive()
                
                # Check if it's binary data
                if "bytes" not in message:
                    # Might be text (configuration update or other message)
                    if "text" in message:
                        try:
                            text_data = json.loads(message["text"])
                            if text_data.get("type") == "config_update":
                                # Update configuration
                                config.update(text_data.get("config", {}))
                                if config.get('debug') or SERVER_LOG_LEVEL == 'DEBUG':
                                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🔧 Config updated: {connection_id}")
                            continue
                        except:
                            pass
                    continue
                
                data = message["bytes"]
                audio_chunk_count += 1
                
                # Log first few chunks only in debug mode (client debug OR server log level DEBUG)
                if (config.get('debug') or SERVER_LOG_LEVEL == 'DEBUG') and audio_chunk_count <= 5:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 📦 Received audio chunk #{audio_chunk_count}: {len(data)} bytes ({connection_id})")
                
                # Convert binary data to numpy array (Float32)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                
                if (config.get('debug') or SERVER_LOG_LEVEL == 'DEBUG') and audio_chunk_count <= 5:
                    rms = np.sqrt(np.mean(audio_chunk.astype(np.float64)**2))
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 📊 Audio chunk: {len(audio_chunk)} samples, RMS: {rms:.6f}, expected: {get_chunk()} samples")
                
                # Process audio and get results
                try:
                    result = recognize_audio_web(
                        audio_chunk=audio_chunk,
                        state=state,
                        config=config
                    )
                    
                    # Send results back to client (send all results, not just chords)
                    if result:
                        await websocket.send_json(result)
                        
                        # Log to server console if enabled
                        if config.get('log') or config.get('debug'):
                            log_result(result, config, connection_id)
                    # Even if result is None, send periodic "listening" updates
                    elif audio_chunk_count % 20 == 0:  # Every ~20 chunks (~0.5 seconds at 44.1kHz)
                        await websocket.send_json({
                            "type": "listening",
                            "timestamp": time.time(),
                            "audio_level": float(np.sqrt(np.mean(audio_chunk.astype(np.float64)**2)))
                        })
                            
                except ValueError as e:
                    # Silently handle low audio level (unless debug mode)
                    if config.get('debug') or SERVER_LOG_LEVEL == 'DEBUG':
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ⚠️  {connection_id}: {str(e)}", file=sys.stderr)
                    # Still send periodic updates even when audio is low
                    if audio_chunk_count % 20 == 0:
                        await websocket.send_json({
                            "type": "listening",
                            "timestamp": time.time(),
                            "message": "Audio level too low"
                        })
                except Exception as e:
                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    print(f"[{timestamp}] ❌ {connection_id}: Error processing audio: {str(e)}", file=sys.stderr)
                    import traceback
                    if config.get('debug') or SERVER_LOG_LEVEL == 'DEBUG':
                        print(f"[{timestamp}] Traceback:", file=sys.stderr)
                        traceback.print_exc(file=sys.stderr)
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
                    
            except Exception as e:
                # Handle WebSocket receive errors
                if isinstance(e, WebSocketDisconnect):
                    raise
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ {connection_id}: Receive error: {str(e)}", file=sys.stderr)
                if config.get('debug') or SERVER_LOG_LEVEL == 'DEBUG':
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                break
                
    except WebSocketDisconnect:
        if SERVER_LOG_LEVEL == 'DEBUG':
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🔌 Disconnected: {connection_id}")
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ Connection error {connection_id}: {str(e)}", file=sys.stderr)
    finally:
        if connection_id in active_connections:
            del active_connections[connection_id]

def log_result(result: dict, config: dict, connection_id: str):
    """Log detection results to server console."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    if result.get('type') == 'frequencies':
        if result.get('frequencies'):
            freq_str = ", ".join([f"{note}({freq:.0f}Hz)" for note, freq in result['frequencies']])
            chroma_str = ""
            if config.get('show_chroma') and result.get('chroma'):
                chroma_values = [f"{val:.3f}" for val in result['chroma']]
                chroma_str = f" | Chroma: [{', '.join(chroma_values)}]"
            print(f"[{timestamp}] [{connection_id}] Frequencies: {freq_str}{chroma_str}")
        else:
            print(f"[{timestamp}] [{connection_id}] No frequencies detected")
    
    elif result.get('type') == 'notes':
        if result.get('notes'):
            note_str = ", ".join([f"{note}({freq:.0f}Hz)" for note, freq in result['notes']])
            chroma_str = ""
            if config.get('show_chroma') and result.get('chroma'):
                chroma_values = [f"{val:.3f}" for val in result['chroma']]
                chroma_str = f" | Chroma: [{', '.join(chroma_values)}]"
            print(f"[{timestamp}] [{connection_id}] Notes: {note_str}{chroma_str}")
        else:
            print(f"[{timestamp}] [{connection_id}] No notes detected")
    
    elif result.get('type') == 'chord':
        if result.get('chord'):
            chord_name = result['chord']
            confidence = result.get('confidence', 0.0)
            stability = result.get('stability', 0)
            stability_suffix = "" if stability >= 2 else " [unstable]"
            
            # Build output to match CLI format exactly
            output_parts = []
            
            # CLI shows chroma if show_chroma is enabled
            if config.get('show_chroma') and result.get('chroma'):
                chroma_values = [f"{val:.3f}" for val in result['chroma']]
                output_parts.append(f"Chroma: [{', '.join(chroma_values)}]")
            
            # CLI shows frequencies if show_frequencies is enabled
            if config.get('show_frequencies') and result.get('notes'):
                note_str = ", ".join([f"{note}({freq:.0f}Hz)" for note, freq in result['notes']])
                output_parts.append(f"Notes: {note_str}")
            
            chord_display = chord_name
            if config.get('debug'):
                chord_display += f" ({confidence:.2f})"
            
            # Match CLI format: "Chroma: [...] -> ChordName [unstable]"
            if output_parts:
                full_output = " | ".join(output_parts) + f" -> {chord_display}"
            else:
                full_output = chord_display
            
            # Remove connection_id from output to match CLI format
            print(f"[{timestamp}] {full_output}{stability_suffix}")
        else:
            if config.get('debug'):
                print(f"[{timestamp}] [{connection_id}] Listening...")
    
    elif result.get('type') == 'listening':
        if config.get('debug'):
            print(f"[{timestamp}] [{connection_id}] 🔇 Listening...")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='FastAPI web server for chord detection')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=9103, help='Port to bind to (default: 9103)')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Logging level for server (default: INFO). Use DEBUG for verbose output.')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes (default: 1). Each worker handles connections independently. '
                            'Note: --reload cannot be used with --workers > 1.')
    return parser.parse_args()

# Global server log level (can be set via --log-level or CHORD_DETECTOR_LOG_LEVEL env var)
# Check environment variable first (for multi-worker support), then default to INFO
SERVER_LOG_LEVEL = os.environ.get("CHORD_DETECTOR_LOG_LEVEL", "INFO").upper()

if __name__ == "__main__":
    import uvicorn
    args = parse_args()
    
    # Validate arguments
    if args.workers > 1 and args.reload:
        print("❌ Error: --reload cannot be used with --workers > 1")
        print("   Use --reload only for development with a single worker")
        sys.exit(1)
    
    # Set log level via environment variable (so all worker processes can access it)
    os.environ["CHORD_DETECTOR_LOG_LEVEL"] = args.log_level.upper()
    # Update module-level variable for current process
    SERVER_LOG_LEVEL = args.log_level.upper()
    
    print(f"🚀 Starting Chord Detector Web Server on {args.host}:{args.port}")
    print(f"📡 WebSocket endpoint: ws://{args.host}:{args.port}/ws")
    print(f"🌐 Web interface: http://{args.host}:{args.port}/")
    if args.workers > 1:
        print(f"⚙️  Multi-process mode: {args.workers} worker processes")
        print(f"   Note: Each worker maintains its own connection state")
    if SERVER_LOG_LEVEL == 'DEBUG':
        print(f"🔍 Debug mode enabled (log level: {SERVER_LOG_LEVEL})")
    
    # Convert log level to lowercase for uvicorn
    uvicorn_log_level = args.log_level.lower()
    
    uvicorn.run(
        "web_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=uvicorn_log_level,
        workers=args.workers if args.workers > 1 else None  # Only set workers if > 1
    )


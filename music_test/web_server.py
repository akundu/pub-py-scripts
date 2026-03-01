"""
FastAPI web server for chord detection with WebSocket audio streaming.
"""
import asyncio
import json
import math
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
    """
    State container for a WebSocket connection.
    Provides the interface expected by the core audio processing function.
    """
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

        # Chord accumulation window for smoothing (reduces rapid chord changes)
        self.chord_accumulator = []  # List of chord detection dicts
        self.chord_window_start = time.time()

        # Get instrument preset if low_freq/high_freq not explicitly set
        instrument = config.get('instrument', 'guitar')
        preset = INSTRUMENT_PRESETS.get(instrument, INSTRUMENT_PRESETS['guitar'])

        self.low_freq = config.get('low_freq') or preset['low_freq']
        self.high_freq = config.get('high_freq') or preset['high_freq']
        self.instrument_name = config.get('instrument_name') or preset['name']
        
        # Song context for chord constraint
        self.song_chords = config.get('song_chords', [])
        self.song_info = config.get('song_info', None)
    
    def update_buffer(self, new_samples, chunk_size):
        """Update circular audio buffer with new samples."""
        buffer_size = len(self.audio_buffer)
        actual_chunk_size = min(chunk_size, len(new_samples))
        end_index = min(self.buffer_index + actual_chunk_size, buffer_size)
        copy_size = end_index - self.buffer_index
        
        if copy_size > 0:
            self.audio_buffer[self.buffer_index:end_index] = new_samples[:copy_size]
        
        if actual_chunk_size > copy_size:
            remaining = actual_chunk_size - copy_size
            self.audio_buffer[:remaining] = new_samples[copy_size:copy_size + remaining]
        
        self.buffer_index = (self.buffer_index + actual_chunk_size) % buffer_size
        return self.buffer_index
    
    def add_detection(self, notes, frequencies, chroma):
        """Add detection results to history."""
        if notes:
            self.notes_history.append(notes)
        if frequencies:
            self.frequencies_history.append(frequencies)
        if chroma is not None:
            self.chroma_history.append(chroma)
    
    def update_chord_stability(self, chord_name):
        """Update chord stability tracking."""
        if chord_name == self.last_chord:
            self.chord_stability += 1
        else:
            self.chord_stability = 0
            self.last_chord = chord_name
        return self.chord_stability
    
    def reset_stability(self):
        """Reset chord stability when no chord detected."""
        self.chord_stability = 0
        self.last_chord = None
    
    def should_log(self, log_interval):
        """Check if enough time has passed for logging."""
        current_time = time.time()
        if current_time - self.last_log_time >= log_interval:
            self.last_log_time = current_time
            return True
        return False
    
    def has_enough_history(self, min_samples=3):
        """Check if we have enough history for progression analysis."""
        return len(self.notes_history) >= min_samples

    def accumulate_chord(self, chord_name, confidence, notes=None, frequencies=None, chroma=None):
        """Add a chord detection to the accumulator for smoothing."""
        self.chord_accumulator.append({
            'chord': chord_name,
            'confidence': confidence,
            'timestamp': time.time(),
            'notes': notes,
            'frequencies': frequencies,
            'chroma': chroma
        })

    def is_window_complete(self, window_duration):
        """Check if the chord accumulation window is complete."""
        elapsed = time.time() - self.chord_window_start
        return elapsed >= window_duration and len(self.chord_accumulator) > 0

    def get_best_chord(self):
        """Get the best chord from accumulated predictions using exponentially-weighted voting with hysteresis."""
        if not self.chord_accumulator:
            return None

        from collections import defaultdict
        DECAY_RATE = self.config.get('decay_rate', 2.3)
        now = time.time()

        chord_scores = defaultdict(lambda: {
            'weighted_confidence': 0.0, 'raw_confidence': 0.0,
            'count': 0, 'best_detection': None
        })

        for detection in self.chord_accumulator:
            chord = detection['chord']
            conf = detection['confidence']
            age = now - detection['timestamp']
            time_weight = math.exp(-DECAY_RATE * age)
            weighted_conf = conf * time_weight

            chord_scores[chord]['weighted_confidence'] += weighted_conf
            chord_scores[chord]['raw_confidence'] += conf
            chord_scores[chord]['count'] += 1

            if (chord_scores[chord]['best_detection'] is None or
                    conf > chord_scores[chord]['best_detection']['confidence']):
                chord_scores[chord]['best_detection'] = detection

        # Hysteresis bonus for current stable chord
        HYSTERESIS_BONUS = self.config.get('hysteresis_bonus', 0.15)
        if self.last_chord and self.chord_stability >= 2:
            if self.last_chord in chord_scores:
                chord_scores[self.last_chord]['weighted_confidence'] *= (1.0 + HYSTERESIS_BONUS)

        best_chord = None
        best_score = 0.0

        for chord, data in chord_scores.items():
            if data['weighted_confidence'] > best_score:
                best_score = data['weighted_confidence']
                best_chord = chord

        if best_chord is None:
            return None

        best_data = chord_scores[best_chord]
        best_detection = best_data['best_detection']

        return {
            'chord': best_chord,
            'confidence': best_data['raw_confidence'] / best_data['count'],
            'votes': best_data['count'],
            'total_votes': len(self.chord_accumulator),
            'notes': best_detection.get('notes'),
            'frequencies': best_detection.get('frequencies'),
            'chroma': best_detection.get('chroma')
        }

    def reset_accumulator(self):
        """Clear the chord accumulator and start a new window."""
        self.chord_accumulator = []
        self.chord_window_start = time.time()

# Store active connections
active_connections: dict[str, ConnectionState] = {}

def parse_config_from_query(query_params: dict) -> dict:
    """Parse configuration from query parameters - match CLI defaults exactly."""
    config = {
        # Core detection parameters
        'silence_threshold': float(query_params.get('silence_threshold', 0.005)),  # Default: 0.005
        'confidence_threshold': float(query_params.get('confidence_threshold', 0.45)),  # Default: 0.45
        'chord_window': float(query_params.get('chord_window', 0.3)),  # Default: 0.3 seconds
        'chord_window_confidence': float(query_params.get('chord_window_confidence', 0.45)),  # Default: 0.45
        # Instrument / frequency range
        'instrument': query_params.get('instrument', 'guitar'),
        'low_freq': None,
        'high_freq': None,
        'instrument_name': None,
        'overlap': float(query_params.get('overlap', 0.75)),  # Default: 0.75
        # Output modes
        'show_frequencies': query_params.get('show_frequencies', 'false').lower() == 'true',
        'show_chroma': query_params.get('show_chroma', 'false').lower() == 'true',
        'frequencies_only': query_params.get('frequencies_only', 'false').lower() == 'true',
        'notes_only': query_params.get('notes_only', 'false').lower() == 'true',
        'debug': query_params.get('debug', 'false').lower() == 'true',
        'log': query_params.get('log', 'false').lower() == 'true',
        'log_interval': float(query_params.get('log_interval', 0.5)),
        # Song constraint
        'song_influence': float(query_params.get('song_influence', 0.7)),
        'map_similar_variants': query_params.get('map_similar_variants', 'true').lower() == 'true',
        # Frequencies/notes-only mode params
        'sensitivity': float(query_params.get('sensitivity', 1.0)),
        'multi_pitch': query_params.get('multi_pitch', 'true').lower() == 'true',
        'show_fft': query_params.get('show_fft', 'false').lower() == 'true',
        'raw_frequencies': query_params.get('raw_frequencies', 'false').lower() == 'true',
        'decay_rate': float(query_params.get('decay_rate', 2.3)),
        'hysteresis_bonus': float(query_params.get('hysteresis_bonus', 0.15)),
    }

    # Apply skill level preset (explicit params above override preset values)
    skill_level = query_params.get('skill_level')
    if skill_level:
        from lib.skill_levels import get_skill_preset, resolve_skill_level, ALL_SKILL_NAMES
        if skill_level in ALL_SKILL_NAMES:
            canonical = resolve_skill_level(skill_level)
            preset = get_skill_preset(canonical)
            for key, value in preset.items():
                # Only apply preset value if the user didn't explicitly set it
                if key not in query_params:
                    config[key] = value
            config['skill_level'] = canonical

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

@app.get("/api/songs")
async def get_songs():
    """Get list of available songs."""
    try:
        from lib.song_loader import get_song_loader
        loader = get_song_loader()
        songs = loader.get_song_list()
        return {
            "songs": songs,
            "total": len(songs)
        }
    except Exception as e:
        return {"error": str(e), "songs": []}

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
            debug_source = []
            if SERVER_LOG_LEVEL == 'DEBUG':
                debug_source.append("server log-level=DEBUG")
            if config.get('debug'):
                debug_source.append("client debug=true")
            debug_note = f" (debug enabled: {', '.join(debug_source)})" if debug_source else ""
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 📊 Configuration{debug_note}:")
            print(f"  Instrument: {config.get('instrument', 'guitar')} ({state.instrument_name})")
            print(f"  Frequency Range: {state.low_freq}-{state.high_freq} Hz")
            print(f"  Silence Threshold: {config.get('silence_threshold', 0.005)}")
            print(f"  Confidence Threshold: {config.get('confidence_threshold', 0.45)}")
            chord_window = config.get('chord_window', 0.3)
            chord_window_confidence = config.get('chord_window_confidence', 0.45)
            print(f"  Chord Window: {chord_window}s {'(smoothing enabled)' if chord_window > 0 else '(instant)'}")
            if chord_window > 0:
                print(f"  Chord Window Confidence: {chord_window_confidence:.3f}")
            print(f"  Overlap: {config.get('overlap', 0.75)} ({get_overlap_ratio()*100:.0f}%, hop size: {get_hop_size()} samples)")
            if config.get('frequencies_only') or config.get('notes_only'):
                mode = 'frequencies_only' if config.get('frequencies_only') else 'notes_only'
                print(f"  Mode: {mode}")
                print(f"  Sensitivity: {config.get('sensitivity', 1.0)}")
                print(f"  Multi-pitch: {config.get('multi_pitch', True)}")
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 📊 Full Config JSON: {json.dumps(config, indent=2)}")
        
        # Send initial acknowledgment (include server log level for client awareness)
        await websocket.send_json({
            "type": "connected",
            "message": f"Connected for {state.instrument_name}",
            "config": config,
            "server_log_level": SERVER_LOG_LEVEL
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
                            elif text_data.get("type") == "set_song":
                                # Load song and update state
                                song_id = text_data.get("song_id")
                                if song_id:
                                    try:
                                        from lib.song_loader import get_song_loader
                                        loader = get_song_loader()
                                        song_chords = loader.get_song_chords(song_id)
                                        song_info = loader.get_song_info(song_id)
                                        
                                        state.song_chords = song_chords
                                        state.song_info = song_info
                                        config['song_chords'] = song_chords
                                        config['song_info'] = song_info
                                        
                                        # Suggest skill level based on song difficulty
                                        suggested_skill = None
                                        if song_info.get('difficulty'):
                                            from lib.skill_levels import suggest_skill_for_song
                                            suggested_skill = suggest_skill_for_song(song_info['difficulty'])

                                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🎵 Song set: {song_info['title']} by {song_info['composer']}")
                                        print(f"   Chords: {', '.join(song_chords)}")

                                        response = {
                                            "type": "song_loaded",
                                            "song_id": song_id,
                                            "song_info": song_info,
                                            "chords": song_chords
                                        }
                                        if suggested_skill:
                                            response["suggested_skill"] = suggested_skill
                                        await websocket.send_json(response)
                                    except Exception as e:
                                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ Error loading song: {e}")
                                        await websocket.send_json({
                                            "type": "error",
                                            "message": f"Failed to load song: {e}"
                                        })
                                else:
                                    # Clear song selection
                                    state.song_chords = []
                                    state.song_info = None
                                    config['song_chords'] = []
                                    config['song_info'] = None
                                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🎵 Song cleared")
                                    await websocket.send_json({
                                        "type": "song_cleared"
                                    })
                            continue
                        except Exception as e:
                            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ Error handling text message: {e}")
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

                    # Get chord window duration (0 = disabled, send immediately)
                    chord_window = config.get('chord_window', 0.0)

                    # Handle chord smoothing window if enabled
                    if chord_window > 0 and result and result.get('type') == 'chord':
                        # Accumulate the chord detection
                        state.accumulate_chord(
                            chord_name=result.get('chord'),
                            confidence=result.get('confidence', 0.0),
                            notes=result.get('notes'),
                            frequencies=result.get('frequencies'),
                            chroma=result.get('chroma')
                        )

                        # Check if window is complete
                        if state.is_window_complete(chord_window):
                            # Get the best chord from accumulated predictions
                            best = state.get_best_chord()
                            # Only output if confidence meets threshold
                            chord_window_confidence = config.get('chord_window_confidence', 0.45)
                            if best and best['confidence'] >= chord_window_confidence:
                                # Build the base result
                                smoothed_result = {
                                    "type": "chord",
                                    "chord": best['chord'],
                                    "confidence": best['confidence'],
                                    "votes": best['votes'],
                                    "total_votes": best['total_votes'],
                                    "stability": state.update_chord_stability(best['chord']),
                                    "timestamp": time.time()
                                }
                                if config.get('show_frequencies') and best.get('notes'):
                                    smoothed_result["notes"] = best['notes']
                                if config.get('show_chroma') and best.get('chroma') is not None:
                                    chroma = best['chroma']
                                    smoothed_result["chroma"] = chroma.tolist() if hasattr(chroma, 'tolist') else chroma
                                
                                # Apply song constraint if enabled
                                if state.song_chords:
                                    from lib.music_understanding import constrain_chord_to_song
                                    chord_result = {
                                        'primary_chords': [best['chord']],
                                        'chord_confidence': best['confidence']
                                    }
                                    constrained = constrain_chord_to_song(
                                        chord_result,
                                        state.song_chords,
                                        song_influence=config.get('song_influence', 0.5),
                                        verbose=config.get('debug')
                                    )
                                    
                                    # Add constrained data to result
                                    smoothed_result['raw_chord'] = constrained['raw_chord']
                                    smoothed_result['raw_confidence'] = constrained['raw_confidence']
                                    smoothed_result['final_chord'] = constrained['final_chord']
                                    smoothed_result['final_confidence'] = constrained['final_confidence']
                                    smoothed_result['song_match'] = constrained['song_match']
                                    smoothed_result['match_type'] = constrained['match_type']
                                    smoothed_result['song_constrained'] = True
                                    if constrained.get('suggested_chord'):
                                        smoothed_result['suggested_chord'] = constrained['suggested_chord']
                                    
                                    # Update stability based on final chord
                                    smoothed_result['stability'] = state.update_chord_stability(constrained['final_chord'])
                                else:
                                    smoothed_result['song_constrained'] = False
                                # Map similar variants (Em/Em7/Emin -> one form) when no song; works with or without song
                                if not state.song_chords and config.get('map_similar_variants', True):
                                    from lib.music_understanding import normalize_chord_variant
                                    smoothed_result['chord'] = normalize_chord_variant(smoothed_result.get('chord') or '')

                                await websocket.send_json(smoothed_result)

                                if config.get('log') or config.get('debug'):
                                    log_result(smoothed_result, config, connection_id)

                            # Reset accumulator for next window
                            state.reset_accumulator()
                        # Don't send individual results when accumulating
                    else:
                        # Send results immediately (chord_window disabled or non-chord result)
                        if result:
                            if result.get('type') == 'chord' and not state.song_chords and config.get('map_similar_variants', True):
                                from lib.music_understanding import normalize_chord_variant
                                result = {**result, 'chord': normalize_chord_variant(result.get('chord') or '')}
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
                    # Silently handle low audio level (only log in DEBUG mode, not ERROR/WARNING/INFO)
                    if SERVER_LOG_LEVEL == 'DEBUG':
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
            # Check if song-constrained
            if result.get('song_constrained'):
                raw_chord = result.get('raw_chord', result['chord'])
                raw_conf = result.get('raw_confidence', result.get('confidence', 0.0))
                final_chord = result.get('final_chord', result['chord'])
                final_conf = result.get('final_confidence', result.get('confidence', 0.0))
                match_indicator = "✓" if result.get('song_match') else "⚠"
                stability = result.get('stability', 0)
                stability_suffix = "" if stability >= 2 else " [unstable]"
                
                # Build output with both raw and final
                raw_str = f"Raw: {raw_chord} ({raw_conf:.1%})"
                final_str = f"→ {final_chord} ({final_conf:.1%}) {match_indicator}"
                
                print(f"[{timestamp}] {raw_str} {final_str}{stability_suffix}")
            else:
                # Non-constrained output (original format)
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


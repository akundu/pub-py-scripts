# file: chord_detector_sounddevice.py
import numpy as np
import time
import argparse
from collections import deque
import sys
from lib.sound_capture import recognize_audio, recognize_audio_with_result
from lib.common import get_hop_size, get_overlap_ratio, set_overlap_ratio, get_buffer_size
from lib.music_understanding import INSTRUMENT_PRESETS
from lib.state import AudioProcessingState

def parse_args():
    parser = argparse.ArgumentParser(description='Sounddevice-based chord detector')
    parser.add_argument('--log', action='store_true', help='Enable logging mode with timestamps')
    parser.add_argument('--log-interval', type=float, default=0.5, help='Logging interval in seconds (default: 0.5)')
    parser.add_argument('--progression', action='store_true', default=True, help='Enable chord progression analysis (default: enabled)')
    parser.add_argument('--sensitivity', type=float, default=0.8, help='Detection sensitivity (0.1-2.0, default: 0.8)')
    parser.add_argument('--silence-threshold', type=float, default=0.005, help='Silence threshold (0-1, default: 0.005)')
    parser.add_argument('--debug', action='store_true', help='Show audio levels for threshold tuning')
    parser.add_argument('--instrument', choices=list(INSTRUMENT_PRESETS.keys()), default='guitar', 
                       help='Instrument preset for frequency filtering (default: guitar)')
    parser.add_argument('--low-freq', type=int, help='Custom low frequency cutoff (Hz)')
    parser.add_argument('--high-freq', type=int, help='Custom high frequency cutoff (Hz)')
    parser.add_argument('--overlap', type=float, default=0.8, help='Overlap ratio (0.0-0.9, default: 0.8)')
    parser.add_argument('--show-frequencies', action='store_true', help='Show detected frequencies')
    parser.add_argument('--show-fft', action='store_true', help='Show autocorrelation analysis data')
    parser.add_argument('--raw-frequencies', action='store_true', help='Use raw frequencies without filtering or processing')
    parser.add_argument('--frequencies-only', action='store_true', help='Show only frequencies, skip chord processing')
    parser.add_argument('--notes-only', action='store_true', help='Show only detected notes, skip chord processing')
    parser.add_argument('--wait-time', type=float, default=0.0, help='Wait time between iterations in seconds (default: 0.0)')
    parser.add_argument('--amplitude-threshold', type=float, default=0.005, help='Minimum amplitude threshold for processing (default: 0.005)')
    parser.add_argument('--show-chroma', action='store_true', help='Show chroma vector along with frequencies')
    parser.add_argument('--single-pitch', action='store_true', help='Use single-pitch detection (autocorrelation only)')
    parser.add_argument('--multi-pitch', action='store_true', default=True, help='Enable multi-pitch detection using FFT (default, better for chords)')
    parser.add_argument('--confidence-threshold', type=float, default=0.2, help='Minimum confidence score to show chords (default: 0.2)')
    parser.add_argument('--chord-window', type=float, default=0.25, help='Chord smoothing window in seconds (0=disabled, default: 0.25)')
    parser.add_argument('--chord-window-confidence', type=float, default=0.485, help='Minimum confidence for chord-window results (default: 0.485)')
    parser.add_argument('--list-devices', action='store_true', help='List available audio input devices and exit')
    parser.add_argument('--device', type=int, help='Audio input device ID (use --list-devices to see available devices)')
    parser.add_argument('--song', type=str, help='Song ID to constrain chord detection (e.g., bg_rk_001)')
    args = parser.parse_args()

    # Handle single-pitch override
    if args.single_pitch:
        args.multi_pitch = False

    return args

# --- 3. Main Application Logic ---
def main():
    """
    Main function to run the sounddevice-based chord detector.
    """
    args = parse_args()

    # Update overlap settings
    set_overlap_ratio(args.overlap)

    # Get instrument settings
    if args.low_freq and args.high_freq:
        low_freq = args.low_freq
        high_freq = args.high_freq
        instrument_name = f"Custom ({low_freq}-{high_freq} Hz)"
    else:
        preset = INSTRUMENT_PRESETS[args.instrument]
        low_freq = preset['low_freq']
        high_freq = preset['high_freq']
        instrument_name = preset['name']

    print(f"🎸 Chord detector listening for {instrument_name}... Press Ctrl+C to stop.")
    
    # Check audio device availability
    try:
        import sounddevice as sd
        
        # List devices if requested
        if args.list_devices:
            print("Available audio input devices:")
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    default_marker = " (DEFAULT)" if i == sd.default.device[0] else ""
                    print(f"  [{i}] {device['name']} - {device['max_input_channels']} channel(s){default_marker}")
            sys.exit(0)
        
        # Use specified device or default
        if args.device is not None:
            sd.default.device = args.device
            print(f"🎤 Using specified device ID: {args.device}")
        
        devices = sd.query_devices()
        default_input = sd.default.device[0]
        if default_input is not None:
            input_device = devices[default_input]
            print(f"🎤 Using input device: {input_device['name']} (ID: {default_input}, channels: {input_device['max_input_channels']})")
            
            # Test audio capture
            print("🔍 Testing audio capture...", end='', flush=True)
            try:
                test_recording = sd.rec(1024, samplerate=44100, channels=1, dtype='float32')
                sd.wait()
                test_rms = np.sqrt(np.mean(test_recording.astype(np.float64)**2))
                if test_rms > 0.0001:
                    print(f" ✓ Audio detected (RMS: {test_rms:.6f})")
                else:
                    print(f" ⚠️  No audio detected (RMS: {test_rms:.6f})")
                    print("   This usually means microphone permissions are not granted.")
                    print("   On macOS: System Settings → Privacy & Security → Microphone")
                    print("   Grant access to Terminal or your Python interpreter")
            except PermissionError:
                print(" ❌ Permission denied!")
                print("   On macOS: System Settings → Privacy & Security → Microphone")
                print("   Grant access to Terminal or your Python interpreter")
                sys.exit(1)
            except Exception as e:
                print(f" ⚠️  Could not test: {e}")
        else:
            print("⚠️  WARNING: No default input device found!")
    except Exception as e:
        print(f"⚠️  Could not query audio devices: {e}")
    
    # Always print all configuration parameters
    print("📊 Configuration Parameters:")
    print(f"  Instrument: {args.instrument} ({instrument_name})")
    if args.low_freq and args.high_freq:
        print(f"  Frequency Range: {low_freq}-{high_freq} Hz (Custom)")
    else:
        print(f"  Frequency Range: {low_freq}-{high_freq} Hz")
    print(f"  Sensitivity: {args.sensitivity}")
    print(f"  Confidence Threshold: {args.confidence_threshold}")
    print(f"  Silence Threshold: {args.silence_threshold}")
    print(f"  Amplitude Threshold: {args.amplitude_threshold}")
    print(f"  Overlap: {args.overlap} ({get_overlap_ratio()*100:.0f}%, hop size: {get_hop_size()} samples)")
    print(f"  Progression: {args.progression}")
    print(f"  Multi-pitch: {getattr(args, 'multi_pitch', True)}")
    print(f"  Single-pitch: {args.single_pitch}")
    print(f"  Show Frequencies: {args.show_frequencies}")
    print(f"  Show Chroma: {getattr(args, 'show_chroma', False)}")
    print(f"  Show FFT: {args.show_fft}")
    print(f"  Raw Frequencies: {args.raw_frequencies}")
    print(f"  Frequencies Only: {args.frequencies_only}")
    print(f"  Notes Only: {args.notes_only}")
    print(f"  Debug: {args.debug}")
    print(f"  Log: {args.log}")
    print(f"  Log Interval: {args.log_interval}s")
    print(f"  Wait Time: {args.wait_time}s")
    chord_window = getattr(args, 'chord_window', 0.0)
    chord_window_confidence = getattr(args, 'chord_window_confidence', 0.485)
    print(f"  Chord Window: {chord_window}s {'(smoothing enabled)' if chord_window > 0 else '(instant)'}")
    if chord_window > 0:
        print(f"  Chord Window Confidence: {chord_window_confidence:.3f} (minimum for window results)")
    
    # Show mode information
    if args.frequencies_only:
        print("📊 FREQUENCIES ONLY MODE: Showing frequencies, skipping chord processing")
    elif args.notes_only:
        print("🎵 NOTES ONLY MODE: Showing detected notes, skipping chord processing")
    else:
        # Default chord detection mode
        print(f"🎯 Confidence threshold: {args.confidence_threshold:.1f} (only showing chords above this score)")
        if args.log:
            print("📝 Logging mode: Showing timestamped chord detections")
    
    if args.debug:
        if getattr(args, 'multi_pitch', True):
            print("🎼 Multi-pitch detection enabled (default, better for chords)")
        else:
            print("🎵 Single-pitch detection enabled (autocorrelation only)")
        if getattr(args, 'show_chroma', False):
            print("🌈 Chroma vector output enabled")
        if args.progression and not args.frequencies_only and not args.notes_only:
            print("📈 Chord progression analysis enabled")

    # Load song data if provided
    song_chords = []
    song_info = None
    if args.song:
        try:
            from lib.song_loader import get_song_loader
            loader = get_song_loader()
            song_chords = loader.get_song_chords(args.song)
            song_info = loader.get_song_info(args.song)
            if song_info:
                print(f"🎵 Loaded song: {song_info['title']} by {song_info['composer']}")
                print(f"   Song chords: {', '.join(song_chords)}")
            else:
                print(f"⚠️  Warning: Song '{args.song}' not found")
        except Exception as e:
            print(f"⚠️  Warning: Could not load song '{args.song}': {e}")

    # Build config dict for the new API
    config = {
        'instrument': args.instrument,
        'sensitivity': args.sensitivity,
        'silence_threshold': args.silence_threshold,
        'amplitude_threshold': args.amplitude_threshold,
        'confidence_threshold': args.confidence_threshold,
        'overlap': args.overlap,
        'progression': args.progression,
        'multi_pitch': getattr(args, 'multi_pitch', True),
        'single_pitch': args.single_pitch,
        'show_frequencies': args.show_frequencies,
        'show_chroma': getattr(args, 'show_chroma', False),
        'show_fft': args.show_fft,
        'raw_frequencies': args.raw_frequencies,
        'frequencies_only': args.frequencies_only,
        'notes_only': args.notes_only,
        'debug': args.debug,
        'log': args.log,
        'log_interval': args.log_interval,
        'chord_window': chord_window,
        'chord_window_confidence': getattr(args, 'chord_window_confidence', 0.485),
        'low_freq': low_freq,
        'high_freq': high_freq,
        'instrument_name': instrument_name,
        'song_chords': song_chords,
        'song_info': song_info,
    }

    # Create state using proper AudioProcessingState class
    state = AudioProcessingState(config)

    # Legacy variables for backward compatibility with recognize_audio
    audio_buffer = state.audio_buffer
    buffer_index = state.buffer_index
    notes_history = state.notes_history
    frequencies_history = state.frequencies_history
    chroma_history = state.chroma_history
    last_chord = state.last_chord
    chord_stability = state.chord_stability
    last_log_time = state.last_log_time

    try:
        if chord_window > 0:
            # Use new chord_window accumulation mode
            print(f"🔄 Chord smoothing enabled: collecting predictions over {chord_window}s windows")
            while True:
                try:
                    result = recognize_audio_with_result(state, config)

                    # If we got a chord result, accumulate it
                    if result and result.get('type') == 'chord':
                        state.accumulate_chord(
                            chord_name=result.get('chord'),
                            confidence=result.get('confidence', 0.0),
                            notes=result.get('notes'),
                            frequencies=result.get('frequencies'),
                            chroma=result.get('chroma')
                        )

                        # Check if window is complete
                        if state.is_window_complete(chord_window):
                            best = state.get_best_chord()
                            # Only output if confidence meets threshold
                            if best and best['confidence'] >= config.get('chord_window_confidence', 0.485):
                                # Apply song constraint if enabled
                                if song_chords:
                                    from lib.music_understanding import constrain_chord_to_song
                                    chord_result = {
                                        'primary_chords': [best['chord']],
                                        'chord_confidence': best['confidence']
                                    }
                                    constrained = constrain_chord_to_song(chord_result, song_chords, verbose=args.debug)
                                    
                                    # Format and print the result
                                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                                    stability = state.update_chord_stability(constrained['final_chord'])
                                    stability_str = "" if stability >= 2 else " [unstable]"
                                    votes_str = f" ({best['votes']}/{best['total_votes']} votes)"
                                    
                                    # Show both raw and constrained
                                    raw_str = f"Raw: {constrained['raw_chord']} ({constrained['raw_confidence']:.1%})"
                                    final_str = f"→ {constrained['final_chord']} ({constrained['final_confidence']:.1%})"
                                    match_indicator = "✓" if constrained['song_match'] else "⚠"
                                    
                                    if args.log:
                                        print(f"[{timestamp}] {raw_str} {final_str} {match_indicator}{votes_str}{stability_str}")
                                    else:
                                        print(f"\r{raw_str} {final_str} {match_indicator}{votes_str}{stability_str}    ", end='', flush=True)
                                else:
                                    # No song context, show raw result
                                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                                    stability = state.update_chord_stability(best['chord'])
                                    stability_str = "" if stability >= 2 else " [unstable]"
                                    votes_str = f" ({best['votes']}/{best['total_votes']} votes)"

                                    if args.log:
                                        print(f"[{timestamp}] {best['chord']} (conf: {best['confidence']:.1%}){votes_str}{stability_str}")
                                    else:
                                        print(f"\r{best['chord']} (conf: {best['confidence']:.1%}){votes_str}{stability_str}    ", end='', flush=True)

                            # Reset for next window
                            state.reset_accumulator()

                    elif result and result.get('type') in ('notes', 'frequencies'):
                        # Pass through non-chord results immediately
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        if result.get('type') == 'notes' and result.get('notes'):
                            notes_str = ", ".join([f"{n[0]}({n[1]:.0f}Hz)" if isinstance(n, (list, tuple)) else str(n) for n in result['notes']])
                            if args.log:
                                print(f"[{timestamp}] Notes: {notes_str}")
                            else:
                                print(f"\rNotes: {notes_str}    ", end='', flush=True)
                        elif result.get('type') == 'frequencies' and result.get('frequencies'):
                            freq_str = ", ".join([f"{f[0]:.0f}Hz" if isinstance(f, (list, tuple)) else str(f) for f in result['frequencies']])
                            if args.log:
                                print(f"[{timestamp}] Frequencies: {freq_str}")
                            else:
                                print(f"\rFrequencies: {freq_str}    ", end='', flush=True)

                except ValueError as e:
                    # Audio level too low - show periodic status
                    current_time = time.time()
                    if not hasattr(state, '_last_status_time'):
                        state._last_status_time = current_time
                    if current_time - state._last_status_time >= 2.0:  # Every 2 seconds
                        if args.debug:
                            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🔇 Audio level too low (threshold: {args.silence_threshold}) - try lowering --silence-threshold", file=sys.stderr)
                        elif args.log:
                            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🔇 Listening... (audio below threshold)")
                        state._last_status_time = current_time
                except Exception as e:
                    print(f"Caught exception: {e}", file=sys.stderr)
                finally:
                    if args.wait_time > 0:
                        time.sleep(args.wait_time)
        else:
            # Use legacy immediate output mode
            iteration_count = 0
            last_status_time = time.time()
            status_interval = 2.0  # Show status every 2 seconds if nothing detected
            
            while True:
                try:
                    recognize_audio(args, audio_buffer, buffer_index, low_freq, high_freq, notes_history, last_chord, chord_stability, last_log_time, frequencies_history, chroma_history, debug=args.debug, frequencies_only=args.frequencies_only, notes_only=args.notes_only, log=args.log)
                    iteration_count += 1
                except ValueError as e:
                    # Audio level too low - show periodic status
                    iteration_count += 1
                    current_time = time.time()
                    if current_time - last_status_time >= status_interval:
                        if args.debug:
                            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🔇 Audio level too low (threshold: {args.silence_threshold}) - try lowering --silence-threshold", file=sys.stderr)
                        elif args.log:
                            # In log mode, show periodic listening status
                            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🔇 Listening... (audio below threshold)")
                        last_status_time = current_time
                except Exception as e:
                    print(f"Caught exception: {e}", file=sys.stderr)
                finally:
                    if args.wait_time > 0:
                        time.sleep(args.wait_time)

    except KeyboardInterrupt:
        print("\n🛑 Stopping sounddevice-based chord detector.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Common reasons for errors:")
        print("1. Microphone not detected or properly configured.")
        print("2. Insufficient permissions (check your OS privacy settings for microphone access).")
        print("3. Another application is already using the microphone.")
        print("4. Missing libraries: Ensure 'sounddevice' and 'scipy' are installed.")

if __name__ == '__main__':
    main() 